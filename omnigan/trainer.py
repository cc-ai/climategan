"""
Main component: the trainer handles everything:
    * initializations
    * training
    * saving
"""
import warnings
from copy import deepcopy
from pathlib import Path
from time import time

from comet_ml import ExistingExperiment
from comet_ml.utils import flatten

warnings.simplefilter("ignore", UserWarning)

import torch
from torch import sigmoid, softmax
import torch.nn as nn
from addict import Dict
from comet_ml import Experiment
from torch import autograd
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from omnigan.classifier import OmniClassifier, get_classifier
from omnigan.data import get_all_loaders
from omnigan.discriminator import OmniDiscriminator, get_dis
from omnigan.eval_metrics import accuracy, mIOU
from omnigan.fid import compute_val_fid
from omnigan.generator import OmniGenerator, get_gen
from omnigan.losses import get_losses
from omnigan.optim import get_optimizer
from omnigan.tutils import (
    divide_pred,
    domains_to_class_tensor,
    fake_domains_to_class_tensor,
    get_num_params,
    get_WGAN_gradient,
    shuffle_batch_tuple,
    vgg_preprocess,
    zero_grad,
    print_num_parameters,
)
from omnigan.utils import (
    comet_kwargs,
    div_dict,
    flatten_opts,
    get_display_indices,
    get_existing_comet_id,
    get_latest_opts,
    merge,
    sum_dict,
)
from omnigan.logger import Logger

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass


class Trainer:
    """Main trainer class"""

    def __init__(self, opts, comet_exp=None, verbose=0, device=None):
        """Trainer class to gather various model training procedures
        such as training evaluating saving and logging

        init:
        * creates an addict.Dict logger
        * creates logger.exp as a comet_exp experiment if `comet` arg is True
        * sets the device (1 GPU or CPU)

        Args:
            opts (addict.Dict): options to configure the trainer, the data, the models
            comet (bool, optional): whether to log the trainer with comet.ml.
                                    Defaults to False.
            verbose (int, optional): printing level to debug. Defaults to 0.
        """
        super().__init__()

        self.opts = opts
        self.verbose = verbose
        self.logger = Logger(self)
        self.loaders = None
        self.losses = None
        self.input_shape = None
        self.G = self.D = self.C = None
        self.lr_names = {}
        self.real_val_fid_stats = None
        self.use_pl4m = False
        self.is_setup = False
        self.current_mode = "train"

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.exp = None
        if isinstance(comet_exp, Experiment):
            self.exp = comet_exp
        self.domain_labels = {"s": 0, "r": 1}

        if self.opts.train.amp:
            optimizers = [
                self.opts.gen.opt.optimizer.lower(),
                self.opts.dis.opt.optimizer.lower(),
                self.opts.classifier.opt.optimizer.lower(),
            ]
            if "extraadam" in optimizers:
                raise ValueError(
                    "AMP does not work with ExtraAdam ({})".format(optimizers)
                )
            self.grad_scaler_d = GradScaler()
            self.grad_scaler_g = GradScaler()
            self.grad_scaler_c = GradScaler()

    def mask(self, z):
        return sigmoid(self.G.decoders["m"](z))

    @torch.no_grad()
    def paint(self, image_batch, mask_batch=None, resolution="approx"):
        """
        Paints a batch of images (or a single image with a batch dim of 1). If
        masks are not provided, they are inferred from the masker.
        Resolution can either be the train-time resolution or the closest
        multiple of 2 ** spade_n_up

        Operations performed without gradient

        If resolution == "approx" then the output image has the shape:
            (dim // 2 ** spade_n_up) * 2 ** spade_n_up, for dim in [height, width]
            eg: (1000, 1300) => (896, 1280) for spade_n_up = 7
        If resolution == "exact" then the output image has the same shape:
            we first process in "approx" mode then upsample bilinear
        If resolution == "basic" image output shape is the train-time's
            (typically 640x640)
        If resolution == "upsample" image is inferred as "basic" and
            then upsampled to original size

        Args:
            image_batch (torch.Tensor): 4D batch of images to flood
            mask_batch (torch.Tensor, optional): Masks for the images.
                Defaults to None (infer with Masker).
            resolution (str, optional): "approx", "exact" or False

        Returns:
            torch.Tensor: N x C x H x W where H and W depend on `resolution`
        """
        assert resolution in {"approx", "exact", "basic", "upsample"}
        previous_mode = self.current_mode
        if previous_mode == "train":
            self.eval_mode()

        if mask_batch is None:
            z = self.G.encode(image_batch)
            mask_batch = self.mask(z)
        else:
            assert len(image_batch) == len(mask_batch)
            assert image_batch.shape[-2:] == mask_batch.shape[-2:]

        z_painter = None
        masked_batch = image_batch * (1.0 - mask_batch)

        if resolution not in {"approx", "exact"}:
            painted = self.G.painter(z_painter, masked_batch)
            if self.opts.gen.p.paste_original_content:
                painted = mask_batch * painted + masked_batch

            if resolution == "upsample":
                painted = nn.functional.interpolate(
                    painted, size=image_batch.shape[-2:], mode="bilinear"
                )
        else:
            # save latent shape
            zh = self.G.painter.z_h
            zw = self.G.painter.z_w
            # adapt latent shape to approximately keep the resolution
            self.G.painter.z_h = (
                image_batch.shape[-2] // 2 ** self.opts.gen.p.spade_n_up
            )
            self.G.painter.z_w = (
                image_batch.shape[-1] // 2 ** self.opts.gen.p.spade_n_up
            )

            painted = self.G.painter(z_painter, masked_batch)
            if self.opts.gen.p.paste_original_content:
                painted = mask_batch * painted + masked_batch

            self.G.painter.z_h = zh
            self.G.painter.z_w = zw
            if resolution == "exact":
                painted = nn.functional.interpolate(
                    painted, size=image_batch.shape[-2:], mode="bilinear"
                )

        if previous_mode == "train":
            self.train_mode()

        return painted

    @classmethod
    def resume_from_path(
        cls, path, overrides={}, setup=True, inference=False, new_exp=False
    ):
        """
        Resume and optionally setup a trainer from a specific path,
        using the latest opts and checkpoint. Requires path to contain opts.yaml
        (or increased), url.txt (or increased) and checkpoints/

        Args:
            path (str | pathlib.Path): Trainer to resume
            overrides (dict, optional): Override loaded opts with those. Defaults to {}.
            setup (bool, optional): Wether or not to setup the trainer before
                returning it. Defaults to True.
            inference (bool, optional): Setup should be done in inference mode or not.
                Defaults to False.
            new_exp (bool, optional): Re-use existing comet exp in path or create
                a new one? Defaults to False.

        Returns:
            omnigan.Trainer: Loaded and resumed trainer
        """
        p = Path(path).expanduser().resolve()
        assert p.exists()

        c = p / "checkpoints"
        assert c.exists() and c.is_dir()

        opts = get_latest_opts(p)
        opts = Dict(merge(overrides, opts))
        opts.train.resume = True

        if new_exp:
            exp = Experiment(project_name="omnigan", **comet_kwargs)
            exp.log_asset_folder(
                str(Path(__file__).parent), recursive=True, log_file_name=True,
            )
            exp.log_parameters(flatten_opts(opts))
        else:
            comet_id = get_existing_comet_id(p)
            exp = ExistingExperiment(previous_experiment=comet_id, **comet_kwargs)

        trainer = cls(opts, comet_exp=exp)
        if setup:
            trainer.setup(inference=inference)
        return trainer

    def save(self):
        save_dir = Path(self.opts.output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "latest_ckpt.pth"

        # Construct relevant state dicts / optims:
        # Save at least G
        save_dict = {
            "epoch": self.logger.epoch,
            "G": self.G.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "step": self.logger.global_step,
        }

        if self.C is not None and get_num_params(self.C) > 0:
            save_dict["C"] = self.C.state_dict()
            save_dict["c_opt"] = self.c_opt.state_dict()
        if self.D is not None and get_num_params(self.D) > 0:
            save_dict["D"] = self.D.state_dict()
            save_dict["d_opt"] = self.d_opt.state_dict()

        if (
            self.logger.epoch >= self.opts.train.min_save_epoch
            and self.logger.epoch % self.opts.train.save_n_epochs == 0
        ):
            torch.save(save_dict, save_dir / f"epoch_{self.logger.epoch}_ckpt.pth")

        torch.save(save_dict, save_path)

    def resume(self, inference=False):
        tpu = "xla" in str(self.device)
        if tpu:
            print("Resuming on TPU:", self.device)

        m_path = Path(self.opts.load_paths.m)
        p_path = Path(self.opts.load_paths.p)
        pm_path = Path(self.opts.load_paths.pm)
        output_path = Path(self.opts.output_path)

        map_loc = self.device if not tpu else "cpu"

        if "m" in self.opts.tasks and "p" in self.opts.tasks:
            # ----------------------------------------
            # -----  Masker and Painter Loading  -----
            # ----------------------------------------

            # want to resume a pm model but no path was provided:
            # resume a single pm model from output_path
            if all([str(p) == "none" for p in [m_path, p_path, pm_path]]):
                checkpoint_path = output_path / "checkpoints/latest_ckpt.pth"
                print("Resuming P+M model from", str(checkpoint_path))
                checkpoint = torch.load(checkpoint_path, map_location=map_loc)

            # want to resume a pm model with a pm_path provided:
            # resume a single pm model from load_paths.pm
            # depending on whether a dir or a file is specified
            elif str(pm_path) != "none":
                assert pm_path.exists()

                if pm_path.is_dir():
                    checkpoint_path = pm_path / "checkpoints/latest_ckpt.pth"
                else:
                    assert pm_path.suffix == ".pth"
                    checkpoint_path = pm_path

                print("Resuming P+M model from", str(checkpoint_path))
                checkpoint = torch.load(checkpoint_path, map_location=map_loc)

            # want to resume a pm model, pm_path not provided:
            # m_path and p_path must be provided as dirs or pth files
            elif m_path != p_path:
                assert m_path.exists()
                assert p_path.exists()

                if m_path.is_dir():
                    m_path = m_path / "checkpoints/latest_ckpt.pth"

                if p_path.is_dir():
                    p_path = p_path / "checkpoints/latest_ckpt.pth"

                assert m_path.suffix == ".pth"
                assert p_path.suffix == ".pth"

                m_checkpoint = torch.load(m_path, map_location=map_loc)
                p_checkpoint = torch.load(p_path, map_location=map_loc)
                checkpoint = merge(m_checkpoint, p_checkpoint)
                print(f"Resuming P+M model from \n  -{p_path} \nand \n  -{m_path}")

            else:
                raise ValueError(
                    "Cannot resume a P+M model with provided load_paths:\n{}".format(
                        self.opts.load_paths
                    )
                )

        else:
            # ----------------------------------
            # -----  Single Model Loading  -----
            # ----------------------------------

            # cannot specify both paths
            if str(m_path) != "none" and str(p_path) != "none":
                raise ValueError(
                    "Opts tasks are {} but received 2 values for the load_paths".format(
                        self.opts.tasks
                    )
                )

            # specified m
            elif str(m_path) != "none":
                assert m_path.exists()
                assert "m" in self.opts.tasks
                model = "M"
                if m_path.is_dir():
                    m_path = m_path / "checkpoints/latest_ckpt.pth"
                checkpoint_path = m_path

            # specified m
            elif str(p_path) != "none":
                assert p_path.exists()
                assert "p" in self.opts.tasks
                model = "P"
                if p_path.is_dir():
                    p_path = p_path / "checkpoints/latest_ckpt.pth"
                checkpoint_path = p_path

            # specified neither p nor m: resume from output_path
            else:
                model = "P" if "p" in self.opts.tasks else "M"
                checkpoint_path = output_path / "checkpoints/latest_ckpt.pth"

            checkpoint = torch.load(checkpoint_path, map_location=map_loc)
            print(f"Resuming {model} model from {checkpoint_path}")

        # On TPUs must send the data to the xla device as it cannot be mapped
        # there directly from torch.load
        if tpu:
            checkpoint = xm.send_cpu_data_to_device(checkpoint, self.device)

        # -----------------------
        # -----  Restore G  -----
        # -----------------------
        self.G.load_state_dict(checkpoint["G"])

        if inference:
            # only G is needed to infer
            print("Done loading checkpoints.")
            return

        self.g_opt.load_state_dict(checkpoint["g_opt"])

        # ------------------------------
        # -----  Resume scheduler  -----
        # ------------------------------
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        for _ in range(self.logger.epoch + 1):
            self.update_learning_rates()

        # Round step to even number for extraGradient
        if self.logger.global_step % 2 != 0:
            self.logger.global_step += 1

        # -----------------------
        # -----  Restore D  -----
        # -----------------------
        if self.D is not None and get_num_params(self.D) > 0:
            self.D.load_state_dict(checkpoint["D"])
            self.d_opt.load_state_dict(checkpoint["d_opt"])

        # -----------------------
        # -----  Restore C  -----
        # -----------------------
        if self.C is not None and get_num_params(self.C) > 0:
            self.C.load_state_dict(checkpoint["C"])
            self.c_opt.load_state_dict(checkpoint["c_opt"])

        # ---------------------------
        # -----  Resore logger  -----
        # ---------------------------
        self.logger.epoch = checkpoint["epoch"]
        self.logger.global_step = checkpoint["step"]
        # Round step to even number for extraGradient
        if self.logger.global_step % 2 != 0:
            self.logger.global_step += 1

    def eval_mode(self):
        """
        Set trainer's models in eval mode
        """
        if self.G is not None:
            self.G.eval()
        if self.D is not None:
            self.D.eval()
        if self.C is not None:
            self.C.eval()
        self.current_mode = "eval"

    def train_mode(self):
        """
        Set trainer's models in train mode
        """
        if self.G is not None:
            self.G.train()
        if self.D is not None:
            self.D.train()
        if self.C is not None:
            self.C.train()
        self.current_mode = "train"

    def assert_z_matches_x(self, x, z):
        assert x.shape[0] == (
            z.shape[0] if not isinstance(z, (list, tuple)) else z[0].shape[0]
        ), "x-> {}, z->{}".format(
            x.shape, z.shape if not isinstance(z, (list, tuple)) else z[0].shape
        )

    def batch_to_device(self, b):
        """sends the data in b to self.device

        Args:
            b (dict): the batch dictionnay

        Returns:
            dict: the batch dictionnary with its "data" field sent to self.device
        """
        for task, tensor in b["data"].items():
            b["data"][task] = tensor.to(self.device)
        return b

    def sample_painter_z(self, batch_size):
        if self.opts.gen.p.no_z:
            return None

        return torch.empty(
            batch_size,
            self.opts.gen.p.latent_dim,
            self.G.painter.z_h,
            self.G.painter.z_w,
            device=self.device,
        ).normal_(mean=0, std=1.0)

    @property
    def train_loaders(self):
        """Get a zip of all training loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["train"].values()))

    @property
    def val_loaders(self):
        """Get a zip of all validation loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["val"].values()))

    def compute_latent_shape(self):
        """Compute the latent shape, i.e. the Encoder's output shape,
        from a batch.

        Raises:
            ValueError: If no loader, the latent_shape cannot be inferred

        Returns:
            tuple: (c, h, w)
        """
        x = None
        for mode in self.loaders:
            for domain in self.loaders[mode]:
                x = self.loaders[mode][domain].dataset[0]["data"]["x"].to(self.device)
                break
            if x is not None:
                break

        if x is None:
            raise ValueError("No batch found to compute_latent_shape")

        x = x.unsqueeze(0)
        z = self.G.encode(x)
        return z.shape[1:] if not isinstance(z, (list, tuple)) else z[0].shape[1:]

    def compute_input_shape(self):
        """Compute the latent shape, i.e. the Encoder's output shape,
        from a batch.

        Raises:
            ValueError: If no loader, the latent_shape cannot be inferred

        Returns:
            tuple: (c, h, w)
        """
        shape = None
        for mode in self.loaders:
            for domain in self.loaders[mode]:
                shape = self.loaders[mode][domain].dataset[0]["data"]["x"].shape
                break
            if shape is not None:
                break

        if shape is None:
            raise ValueError("No batch found to compute_latent_shape")

        return shape

    def g_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.gen.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.g_opt.extrapolation()
        else:
            self.g_opt.step()

    def d_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.dis.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.d_opt.extrapolation()
        else:
            self.d_opt.step()

    def c_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.classifier.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.c_opt.extrapolation()
        else:
            self.c_opt.step()

    def update_learning_rates(self):
        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()
        if self.c_scheduler is not None:
            self.c_scheduler.step()

    def setup(self, inference=False):
        """Prepare the trainer before it can be used to train the models:
        * initialize G and D
        * compute latent space dims and create classifier accordingly
        * creates 3 optimizers
        """
        self.logger.global_step = 0
        start_time = time()
        self.logger.time.start_time = start_time
        verbose = self.verbose

        if not inference:
            self.loaders = get_all_loaders(self.opts)

        # -----------------------
        # -----  Generator  -----
        # -----------------------
        __t = time()
        print("Creating generator:")
        self.G: OmniGenerator = get_gen(self.opts, verbose=verbose, no_init=inference)
        print("Sending to", self.device)
        self.G = self.G.to(self.device)
        print(f"Generator OK in {time() - __t:.1f}s.")

        if self.input_shape is None:
            if inference:
                raise ValueError(
                    "Cannot auto-set input_shape from loaders in inference mode."
                    + " It  has to  be set prior to setup()."
                )
            print("Computing latent & input shapes...", end="", flush=True)
            self.input_shape = self.compute_input_shape()

        if "s" in self.opts.tasks:
            self.G.decoders["s"].set_target_size(self.input_shape[-2:])
        print("OK.")

        self.G.painter.z_h = self.input_shape[-2] // (2 ** self.opts.gen.p.spade_n_up)
        self.G.painter.z_w = self.input_shape[-1] // (2 ** self.opts.gen.p.spade_n_up)

        if inference:
            print("Inference mode: no Discriminator, no Classifier, no optimizers")
            print_num_parameters(self)
            return

        # ---------------------------
        # -----  Discriminator  -----
        # ---------------------------

        self.D: OmniDiscriminator = get_dis(self.opts, verbose=verbose).to(self.device)
        print("Discriminator OK.")

        # ------------------------
        # -----  Classifier  -----
        # ------------------------

        self.C: OmniClassifier = None
        if self.G.encoder is not None and self.opts.train.latent_domain_adaptation:
            self.latent_shape = self.compute_latent_shape()
            self.C = get_classifier(self.opts, self.latent_shape, verbose=verbose).to(
                self.device
            )
            print("Classifier OK.")

        print_num_parameters(self)

        # --------------------------
        # -----  Optimization  -----
        # --------------------------
        # Get different optimizers for each task (different learning rates)
        self.g_opt, self.g_scheduler, self.lr_names["G"] = get_optimizer(
            self.G, self.opts.gen.opt, self.opts.tasks
        )

        if get_num_params(self.D) > 0:
            self.d_opt, self.d_scheduler, self.lr_names["D"] = get_optimizer(
                self.D, self.opts.dis.opt, self.opts.tasks
            )
        else:
            self.d_opt, self.d_scheduler = None, None

        if self.C is not None:
            self.c_opt, self.c_scheduler, self.lr_names["C"] = get_optimizer(
                self.C, self.opts.classifier.opt, None
            )
        else:
            self.c_opt, self.c_scheduler = None, None

        if self.opts.train.resume:
            self.resume()

        self.losses = get_losses(self.opts, verbose, device=self.device)

        if verbose > 0:
            for mode, mode_dict in self.loaders.items():
                for domain, domain_loader in mode_dict.items():
                    print(
                        "Loader {} {} : {}".format(
                            mode, domain, len(domain_loader.dataset)
                        )
                    )

        # ----------------------------
        # -----  Display images  -----
        # ----------------------------
        self.display_images = {}
        for mode, mode_dict in self.loaders.items():
            self.display_images[mode] = {}
            for domain, domain_loader in mode_dict.items():
                dataset = self.loaders[mode][domain].dataset
                display_indices = get_display_indices(self.opts, domain, len(dataset))
                ldis = len(display_indices)
                print(
                    f"Creating {ldis} {mode} {domain} display images...",
                    end="\r",
                    flush=True,
                )
                self.display_images[mode][domain] = [
                    Dict(dataset[i]) for i in display_indices if i < len(dataset)
                ]
                if self.exp is not None:
                    for im_id, d in enumerate(self.display_images[mode][domain]):
                        self.exp.log_parameter(
                            "display_image_{}_{}_{}".format(mode, domain, im_id),
                            d["paths"],
                        )
        print(" " * 50, end="\r")
        print("Done creating display images")
        print("Setup done.")
        self.is_setup = True

    def train(self):
        """For each epoch:
        * train
        * eval
        * save
        """
        assert self.is_setup

        for self.logger.epoch in range(
            self.logger.epoch, self.logger.epoch + self.opts.train.epochs
        ):
            self.run_epoch()
            self.run_evaluation(verbose=1)
            self.save()

            if self.logger.epoch == self.opts.gen.p.pl4m_epoch:
                self.use_pl4m = True

    def run_epoch(self):
        """Runs an epoch:
        * checks trainer is setup
        * gets a tuple of batches per domain
        * sends batches to device
        * updates sequentially G, D, C
        """
        assert self.is_setup
        self.train_mode()
        if self.exp is not None:
            self.exp.log_parameter("epoch", self.logger.epoch)
        epoch_len = min(len(loader) for loader in self.loaders["train"].values())
        epoch_desc = "Epoch {}".format(self.logger.epoch)
        self.logger.time.epoch_start = time()

        for multi_batch_tuple in tqdm(
            self.train_loaders,
            desc=epoch_desc,
            total=epoch_len,
            mininterval=0.5,
            unit="batch",
        ):

            self.logger.time.step_start = time()
            multi_batch_tuple = shuffle_batch_tuple(multi_batch_tuple)

            # The `[0]` is because the domain is contained in a list
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }

            # ------------------------------
            # -----  Update Generator  -----
            # ------------------------------

            # freeze params of the discriminator
            if self.d_opt is not None:
                for param in self.D.parameters():
                    param.requires_grad = False

            self.update_G(multi_domain_batch)

            # ----------------------------------
            # -----  Update Discriminator  -----
            # ----------------------------------

            # unfreeze params of the discriminator
            if self.d_opt is not None:
                for param in self.D.parameters():
                    param.requires_grad = True

                self.update_D(multi_domain_batch)

            # -------------------------------
            # -----  Update Classifier  -----
            # -------------------------------
            if self.opts.train.latent_domain_adaptation and self.C is not None:
                self.update_C(multi_domain_batch)

            # -------------------------
            # -----  Log Metrics  -----
            # -------------------------
            self.logger.global_step += 1
            self.logger.log_step_time(time())

        self.update_learning_rates()
        self.logger.log_learning_rates()
        self.logger.log_epoch_time(time())

    def update_G(self, multi_domain_batch, verbose=0):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

        * automatic mixed precision according to self.opts.train.amp
        * compute loss for each task
        * loss.backward()
        * g_opt_step()
            * g_opt.step() or .extrapolation() depending on self.logger.global_step
        * logs losses on comet.ml with self.logger.log_losses(model_to_update="G")

        Args:
            multi_domain_batch (dict): dictionnary of domain batches
        """
        zero_grad(self.G)
        if self.opts.train.amp:
            with autocast():
                g_loss = self.get_G_loss(multi_domain_batch, verbose)
            self.grad_scaler_g.scale(g_loss).backward()
            self.grad_scaler_g.step(self.g_opt)
            self.grad_scaler_g.update()
        else:
            g_loss = self.get_G_loss(multi_domain_batch, verbose)
            g_loss.backward()
            self.g_opt_step()

        self.logger.log_losses(model_to_update="G", mode="train")

    def update_D(self, multi_domain_batch, verbose=0):
        zero_grad(self.D)

        if self.opts.train.amp:
            with autocast():
                d_loss = self.get_D_loss(multi_domain_batch, verbose)
            self.grad_scaler_d.scale(d_loss).backward()
            self.grad_scaler_d.step(self.d_opt)
            self.grad_scaler_d.update()
        else:
            d_loss = self.get_D_loss(multi_domain_batch, verbose)
            d_loss.backward()
            self.d_opt_step()

        self.logger.losses.disc.total_loss = d_loss.item()
        self.logger.log_losses(model_to_update="D", mode="train")

    def update_C(self, multi_domain_batch):
        """
        Update the classifier using normal labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
                the trainer's loaders

        """
        zero_grad(self.C)
        if self.opts.train.amp:
            with autocast():
                c_loss = self.get_C_loss(multi_domain_batch)
            self.grad_scaler_c.scale(c_loss).backward()
            self.grad_scaler_c.step(self.c_opt)
            self.grad_scaler_c.update()
        else:
            c_loss = self.get_C_loss(multi_domain_batch)
            self.logger.losses.classifier = c_loss.item()
            c_loss.backward()
            self.c_opt_step()

        self.logger.losses.classifier = c_loss.item()

    def get_C_loss(self, multi_domain_batch):
        """Compute the loss of the domain classifier with real labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas.C
        """
        loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain == "rf":
                continue
            z = self.G.encode(batch["data"]["x"])
            # Forward through classifier, output classifier = (batch_size, 4)
            output_classifier = self.C(z)
            # Cross entropy loss (with sigmoid)
            update_loss = self.losses["C"](
                output_classifier,
                domains_to_class_tensor(batch["domain"], one_hot).to(self.device),
            )
            loss += update_loss

        return lambdas.C * loss

    def get_D_loss(self, multi_domain_batch, verbose=0):
        """Compute the discriminators' losses:

        * for each domain-specific batch:
        * encode the image
        * get the conditioning tensor if using spade
        * source domain is the data's domain, sequentially r|s then f|n
        * get the target domain accordingly
        * compute the translated image from the data
        * compute the source domain discriminator's loss on the data
        * compute the target domain discriminator's loss on the translated image

        # ? In this setting, each D[decoder][domain] is updated twice towards
        # real or fake data

        See readme's update d section for details

        Args:
            multi_domain_batch ([type]): [description]

        Returns:
            [type]: [description]
        """

        disc_loss = {
            "m": {"Advent": 0},
            "s": {"Advent": 0},
        }
        if self.opts.dis.p.use_local_discriminator:
            disc_loss["p"] = {"global": 0, "local": 0}
        else:
            disc_loss["p"] = {"gan": 0}

        for domain, batch in multi_domain_batch.items():
            x = batch["data"]["x"]

            # ---------------------
            # -----  Painter  -----
            # ---------------------
            if domain == "rf":
                m = batch["data"]["m"]
                # sample vector
                with torch.no_grad():
                    # see spade compute_discriminator_loss
                    z_paint = self.sample_painter_z(x.shape[0])
                    fake = self.G.painter(z_paint, x * (1.0 - m))
                    if self.opts.gen.p.paste_original_content:
                        fake = fake * m + x * (1.0 - m)
                    fake = fake.detach()
                    fake.requires_grad_()

                if self.opts.dis.p.use_local_discriminator:
                    fake_d_global = self.D["p"]["global"](fake)
                    real_d_global = self.D["p"]["global"](x)

                    fake_d_local = self.D["p"]["local"](fake * m)
                    real_d_local = self.D["p"]["local"](x * m)

                    global_loss = self.losses["D"]["p"](fake_d_global, False, True)
                    global_loss += self.losses["D"]["p"](real_d_global, True, True)

                    local_loss = self.losses["D"]["p"](fake_d_local, False, True)
                    local_loss += self.losses["D"]["p"](real_d_local, True, True)

                    disc_loss["p"]["global"] += global_loss
                    disc_loss["p"]["local"] += local_loss
                else:
                    real_cat = torch.cat([m, x], axis=1)
                    fake_cat = torch.cat([m, fake], axis=1)
                    real_fake_cat = torch.cat([real_cat, fake_cat], dim=0)
                    real_fake_d = self.D["p"](real_fake_cat)
                    real_d, fake_d = divide_pred(real_fake_d)
                    disc_loss["p"]["gan"] = self.losses["D"]["p"](fake_d, False, True)
                    disc_loss["p"]["gan"] += self.losses["D"]["p"](real_d, True, True)

            # --------------------
            # -----  Masker  -----
            # --------------------
            else:
                z = self.G.encode(x)
                for task, _ in batch["data"].items():
                    if task == "m":
                        step_loss = self.masker_m_loss(x, z, None, domain, for_="D")
                        step_loss *= self.opts.train.lambdas.advent.adv_main
                        disc_loss["m"]["Advent"] += step_loss

                    if task == "s":
                        step_loss = self.masker_s_loss(x, z, None, domain, for_="D")
                        step_loss *= self.opts.train.lambdas.advent.adv_main
                        disc_loss["s"]["Advent"] += step_loss

        self.logger.losses.disc.update(
            {
                dom: {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                }
                for dom, d in disc_loss.items()
            }
        )

        loss = sum(v for d in disc_loss.values() for k, v in d.items())
        return loss

    def get_G_loss(self, multi_domain_batch, verbose=0):
        m_loss = p_loss = None

        # For now, always compute "representation loss"
        g_loss = 0

        if any(t in self.opts.tasks for t in "msd"):
            m_loss = self.get_masker_loss(multi_domain_batch)
            self.logger.losses.gen.masker = m_loss.item()
            g_loss += m_loss

        if "p" in self.opts.tasks:
            p_loss = self.get_painter_loss(multi_domain_batch)
            self.logger.losses.gen.painter = p_loss.item()
            g_loss += p_loss

        assert g_loss != 0 and not isinstance(g_loss, int), "No update in get_G_loss!"

        self.logger.losses.gen.total_loss = g_loss.item()

        return g_loss

    def get_masker_loss(self, multi_domain_batch):  # TODO update docstrings
        """Only update the representation part of the model, meaning everything
        but the translation part

        * for each batch in available domains:
            * compute latent classifier loss with fake labels(1)
            * compute task-specific losses (2)
            * compute the adaptation and translation decoders' auto-encoding losses (3)
            * compute the adaptation decoder's translation losses (GAN and Cycle) (4)

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        m_loss = 0
        for domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if domain == "rf":
                continue

            x = batch["data"]["x"]
            z = self.G.encode(x)
            # ---------------------------------
            # -----  classifier loss (1)  -----
            # ---------------------------------
            if self.opts.train.latent_domain_adaptation:
                loss = self.masker_c_loss(z, batch["domain"])
                m_loss += loss
                self.logger.losses.gen.classifier[domain] = loss.item()

            # --------------------------------------
            # -----  task-specific losses (2)  -----
            # --------------------------------------
            for task, target in batch["data"].items():
                if task == "m":
                    loss = self.masker_m_loss(x, z, target, domain, "G")
                    m_loss += loss
                    self.logger.losses.gen.task["m"][domain] = loss.item()
                elif task == "s":
                    loss = self.masker_s_loss(x, z, target, domain, "G")
                    m_loss += loss
                    self.logger.losses.gen.task["s"][domain] = loss.item()
                elif task == "d":
                    loss = self.masker_d_loss(x, z, target, domain, "G")
                    m_loss += loss
                    self.logger.losses.gen.task["d"][domain] = loss.item()

        return m_loss

    def get_painter_loss(self, multi_domain_batch):
        """Computes the translation loss when flooding/deflooding images

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        step_loss = 0
        # self.g_opt.zero_grad()
        lambdas = self.opts.train.lambdas
        batch_domain = "rf"
        batch = multi_domain_batch[batch_domain]

        x = batch["data"]["x"]
        # ! different mask: hides water to be reconstructed
        # ! 1 for water, 0 otherwise
        m = batch["data"]["m"]
        z = self.sample_painter_z(x.shape[0])
        masked_x = x * (1.0 - m)

        fake_flooded = self.G.painter(z, masked_x)
        if self.opts.gen.p.paste_original_content:
            fake_flooded = masked_x + m * fake_flooded

        # ----------------------
        # -----  VGG Loss  -----
        # ----------------------
        if lambdas.G.p.vgg != 0:
            loss = self.losses["G"]["p"]["vgg"](
                vgg_preprocess(fake_flooded * m), vgg_preprocess(x * m)
            )
            loss *= lambdas.G.p.vgg
            self.logger.losses.gen.p.vgg = loss.item()
            step_loss += loss

        # ---------------------
        # -----  TV Loss  -----
        # ---------------------
        if lambdas.G.p.tv != 0:
            loss = self.losses["G"]["p"]["tv"](fake_flooded * m)
            loss *= lambdas.G.p.tv
            self.logger.losses.gen.p.tv = loss.item()
            step_loss += loss

        # --------------------------
        # -----  Context Loss  -----
        # --------------------------
        if lambdas.G.p.context != 0:
            loss = self.losses["G"]["p"]["context"](fake_flooded, x, m)
            loss *= lambdas.G.p.context
            self.logger.losses.gen.p.context = loss.item()
            step_loss += loss

        # ---------------------------------
        # -----  Reconstruction Loss  -----
        # ---------------------------------
        if lambdas.G.p.reconstruction != 0:
            loss = self.losses["G"]["p"]["reconstruction"](fake_flooded, x, m)
            loss *= lambdas.G.p.reconstruction
            self.logger.losses.gen.p.reconstruction = loss.item()
            step_loss += loss

        # -------------------------------------
        # -----  Local & Global GAN Loss  -----
        # -------------------------------------
        if self.opts.dis.p.use_local_discriminator:
            fake_d_global = self.D["p"]["global"](fake_flooded)
            fake_d_local = self.D["p"]["local"](fake_flooded * m)

            real_d_global = self.D["p"]["global"](x)

            # Note: discriminator returns [out_1,...,out_num_D] outputs
            # Each out_i is a list [feat1, feat2, ..., pred_i]

            self.logger.losses.gen.p.gan = 0

            loss = self.losses["G"]["p"]["gan"](fake_d_global, True, False)
            loss += self.losses["G"]["p"]["gan"](fake_d_local, True, False)
            loss *= lambdas.G["p"]["gan"]

            self.logger.losses.gen.p.gan = loss.item()

            step_loss += loss

            # -----------------------------------
            # -----  Feature Matching Loss  -----
            # -----------------------------------
            # (only on global discriminator)
            # Order must be real, fake
            if self.opts.dis.p.get_intermediate_features:
                loss = self.losses["G"]["p"]["featmatch"](real_d_global, fake_d_global)
                loss *= lambdas.G["p"]["featmatch"]

                if isinstance(loss, float):
                    self.logger.losses.gen.p.featmatch = loss
                else:
                    self.logger.losses.gen.p.featmatch = loss.item()

                step_loss += loss

        # -------------------------------------------
        # -----  Single Discriminator GAN Loss  -----
        # -------------------------------------------
        else:
            real_cat = torch.cat([m, x], axis=1)
            fake_cat = torch.cat([m, fake_flooded], axis=1)
            real_fake_cat = torch.cat([real_cat, fake_cat], dim=0)

            real_fake_d = self.D["p"](real_fake_cat)
            real_d, fake_d = divide_pred(real_fake_d)

            loss = self.losses["G"]["p"]["gan"](fake_d, True, False)
            self.logger.losses.gen.p.gan = loss.item()
            step_loss += loss

            # -----------------------------------
            # -----  Feature Matching Loss  -----
            # -----------------------------------
            if self.opts.dis.p.get_intermediate_features and lambdas.G.p.featmatch != 0:
                loss = self.losses["G"]["p"]["featmatch"](real_d, fake_d)
                loss *= lambdas.G.p.featmatch

                if isinstance(loss, float):
                    self.logger.losses.gen.p.featmatch = loss
                else:
                    self.logger.losses.gen.p.featmatch = loss.item()

            step_loss += loss

        return step_loss

    def masker_c_loss(self, z, target, for_="G"):
        assert for_ in {"G", "D"}
        full_loss = torch.tensor(0.0, device=self.device)
        # -------------------
        # -----  Depth  -----
        # -------------------
        one_hot = self.opts.classifier.loss != "cross_entropy"
        output_classifier = self.C(z)
        # Cross entropy loss (with sigmoid) with fake labels to fool C
        loss = self.losses["G"]["classifier"](
            output_classifier, fake_domains_to_class_tensor(target, one_hot),
        )
        loss *= self.opts.train.lambdas.G.classifier
        full_loss += loss

        return full_loss

    def masker_d_loss(self, x, z, target, domain, for_="G"):
        assert for_ in {"G", "D"}
        self.assert_z_matches_x(x, z)
        assert x.shape[0] == target.shape[0]
        full_loss = torch.tensor(0.0, device=self.device)
        weight = self.opts.train.lambdas.G.d.main

        if weight == 0:
            return full_loss

        if domain == "r" and not self.opts.gen.d.use_pseudo_labels:
            return full_loss

        # -------------------
        # -----  Depth  -----
        # -------------------

        prediction = self.G.decoders["d"](z)
        loss = self.losses["G"]["tasks"]["d"](prediction, target)
        loss *= weight

        full_loss += loss

        return full_loss

    def masker_s_loss(self, x, z, target, domain, for_="G"):
        assert for_ in {"G", "D"}
        assert domain in {"r", "s"}
        self.assert_z_matches_x(x, z)
        assert x.shape[0] == target.shape[0] if target is not None else True
        full_loss = torch.tensor(0.0, device=self.device)
        softmax_preds = None
        # --------------------------
        # -----  Segmentation  -----
        # --------------------------
        pred = None
        if for_ == "G" or self.opts.gen.s.use_advent:
            pred = self.G.decoders["s"](z)

        # Supervised segmentation loss: crossent for sim domain,
        # crossent_pseudo for real ; loss is crossent in any case
        if for_ == "G":
            if domain == "s" or self.opts.gen.s.use_pseudo_labels:
                if domain == "s":
                    logger = self.logger.losses.gen.task["s"]["crossent"]
                    weight = self.opts.train.lambdas.G["s"]["crossent"]
                else:
                    logger = self.logger.losses.gen.task["s"]["crossent_pseudo"]
                    weight = self.opts.train.lambdas.G["s"]["crossent_pseudo"]

                if weight != 0:
                    # Cross-Entropy loss
                    loss_func = self.losses["G"]["tasks"]["s"]["crossent"]
                    loss = loss_func(pred, target.squeeze(1))
                    loss *= weight
                    full_loss += loss
                    logger[domain] = loss.item()

            if domain == "r":
                weight = self.opts.train.lambdas.G["s"]["minent"]
                if self.opts.gen.s.use_minent and weight != 0:
                    softmax_preds = softmax(pred, dim=1)
                    # Entropy minimization loss
                    loss = self.losses["G"]["tasks"]["s"]["minent"](softmax_preds)
                    loss *= weight
                    full_loss += loss

                    self.logger.losses.gen.task["s"]["minent"]["r"] = loss.item()

        # Fool ADVENT discriminator
        if self.opts.gen.s.use_advent:
            if for_ == "D":
                domain_label = domain
                logger = {}
                loss_func = self.losses["D"]["advent"]
                pred = pred.detach()
                weight = self.opts.train.lambdas.advent.adv_main
            else:
                domain_label = "s"
                logger = self.logger.losses.gen.task["s"]["advent"]
                loss_func = self.losses["G"]["tasks"]["s"]["advent"]
                weight = self.opts.train.lambdas.G["s"]["advent"]

            if (for_ == "D" or domain == "r") and weight != 0:
                if softmax_preds is None:
                    softmax_preds = softmax(pred, dim=1)
                loss = loss_func(
                    softmax_preds,
                    self.domain_labels[domain_label],
                    self.D["s"]["Advent"],
                )
                loss *= weight
                full_loss += loss
                logger[domain] = loss.item()

                if for_ == "D":
                    # WGAN: clipping or GP
                    if self.opts.dis.s.gan_type == "GAN" or "WGAN_norm":
                        pass
                    elif self.opts.dis.s.gan_type == "WGAN":
                        for p in self.D["s"]["Advent"].parameters():
                            p.data.clamp_(
                                self.opts.dis.s.wgan_clamp_lower,
                                self.opts.dis.s.wgan_clamp_upper,
                            )
                    elif self.opts.dis.s.gan_type == "WGAN_gp":
                        prob_need_grad = autograd.Variable(pred, requires_grad=True)
                        d_out = self.D["s"]["Advent"](prob_need_grad)
                        gp = get_WGAN_gradient(prob_need_grad, d_out)
                        gp_loss = gp * self.opts.train.lambdas.advent.WGAN_gp
                        full_loss += gp_loss
                    else:
                        raise NotImplementedError

        return full_loss

    def masker_m_loss(self, x, z, target, domain, for_="G"):
        assert for_ in {"G", "D"}
        assert domain in {"r", "s"}
        self.assert_z_matches_x(x, z)
        assert x.shape[0] == target.shape[0] if target is not None else True
        full_loss = torch.tensor(0.0, device=self.device)
        # ? output features classifier
        pred_logits = self.G.decoders["m"](z)
        pred_prob = sigmoid(pred_logits)
        pred_prob_complementary = 1 - pred_prob
        prob = torch.cat([pred_prob, pred_prob_complementary], dim=1)

        if for_ == "G":
            # TV loss
            weight = self.opts.train.lambdas.G.m.tv
            if weight != 0:
                loss = self.losses["G"]["tasks"]["m"]["tv"](pred_prob)
                loss *= weight
                full_loss += loss

                self.logger.losses.gen.task["m"]["tv"][domain] = loss.item()

            weight = self.opts.train.lambdas.G.m.bce
            if domain == "s" and weight != 0:
                # CrossEnt Loss
                loss = self.losses["G"]["tasks"]["m"]["bce"](pred_logits, target)
                loss *= weight
                full_loss += loss
                self.logger.losses.gen.task["m"]["bce"]["s"] = loss.item()

            if domain == "r":

                weight = self.opts.train.lambdas.G["m"]["gi"]
                if self.opts.gen.m.use_ground_intersection and weight != 0:
                    if self.verbose > 0:
                        print("Using GroundIntersection loss.")
                    # GroundIntersection loss
                    loss = self.losses["G"]["tasks"]["m"]["gi"](pred_prob, target)
                    loss *= weight
                    full_loss += loss
                    self.logger.losses.gen.task["m"]["gi"]["r"] = loss.item()

                weight = self.opts.train.lambdas.G.m.pl4m
                if self.use_pl4m and weight != 0:
                    # Painter loss
                    pl4m_loss = self.painter_loss_for_masker(x, pred_prob)
                    pl4m_loss *= weight
                    full_loss += pl4m_loss
                    self.logger.losses.gen.task.m.pl4m.r = pl4m_loss.item()

                weight = self.opts.train.lambdas.advent.ent_main
                if self.opts.gen.m.use_minent and weight != 0:
                    # MinEnt loss
                    loss = self.losses["G"]["tasks"]["m"]["minent"](prob)
                    loss *= weight
                    full_loss += loss
                    self.logger.losses.gen.task["m"]["minent"]["r"] = loss.item()

        if self.opts.gen.m.use_advent:
            # AdvEnt loss
            if for_ == "D":
                domain_label = domain
                logger = {}
                loss_func = self.losses["D"]["advent"]
                prob = prob.detach()
                weight = self.opts.train.lambdas.advent.adv_main
            else:
                domain_label = "s"
                logger = self.logger.losses.gen.task["m"]["advent"]
                loss_func = self.losses["G"]["tasks"]["m"]["advent"]
                weight = self.opts.train.lambdas.advent.adv_main

            if (for_ == "D" or domain == "r") and weight != 0:
                loss = loss_func(
                    prob.to(self.device),
                    self.domain_labels[domain_label],
                    self.D["m"]["Advent"],
                )
                loss *= weight
                full_loss += loss
                logger[domain] = loss.item()

            if for_ == "D":
                # WGAN: clipping or GP
                if self.opts.dis.m.gan_type == "GAN" or "WGAN_norm":
                    pass
                elif self.opts.dis.m.gan_type == "WGAN":
                    for p in self.D["s"]["Advent"].parameters():
                        p.data.clamp_(
                            self.opts.dis.m.wgan_clamp_lower,
                            self.opts.dis.m.wgan_clamp_upper,
                        )
                elif self.opts.dis.m.gan_type == "WGAN_gp":
                    prob_need_grad = autograd.Variable(prob, requires_grad=True)
                    d_out = self.D["s"]["Advent"](prob_need_grad)
                    gp = get_WGAN_gradient(prob_need_grad, d_out)
                    gp_loss = self.opts.train.lambdas.advent.WGAN_gp * gp
                    full_loss += gp_loss
                else:
                    raise NotImplementedError

        return full_loss

    def painter_loss_for_masker(self, x, m):
        # pl4m loss
        # painter should not be updated
        for param in self.G.painter.parameters():
            param.requires_grad = False

        z = self.sample_painter_z(x.shape[0])
        masked_x = x * (1.0 - m)  # 0s where water should be painted
        fake_flooded = self.G.painter(z, masked_x)
        if self.opts.gen.p.paste_original_content:
            fake_flooded = masked_x + m * fake_flooded

        if self.opts.dis.p.use_local_discriminator:
            fake_d_global = self.D["p"]["global"](fake_flooded)
            fake_d_local = self.D["p"]["local"](fake_flooded * m)

            # Note: discriminator returns [out_1,...,out_num_D] outputs
            # Each out_i is a list [feat1, feat2, ..., pred_i]

            pl4m_loss = self.losses["G"]["p"]["gan"](fake_d_global, True, False)
            pl4m_loss += self.losses["G"]["p"]["gan"](fake_d_local, True, False)
        else:
            real_cat = torch.cat([m, x], axis=1)
            fake_cat = torch.cat([m, fake_flooded], axis=1)
            real_fake_cat = torch.cat([real_cat, fake_cat], dim=0)

            real_fake_d = self.D["p"](real_fake_cat)
            _, fake_d = divide_pred(real_fake_d)

            pl4m_loss = self.losses["G"]["p"]["gan"](fake_d, True, False)

        if "p" in self.opts.tasks:
            for param in self.G.painter.parameters():
                param.requires_grad = True

        return pl4m_loss

    @torch.no_grad()
    def run_evaluation(self, verbose=0):
        print("******************* Running Evaluation ***********************")
        start_time = time()
        self.eval_mode()
        val_logger = None
        nb_of_batches = None
        for i, multi_batch_tuple in enumerate(self.val_loaders):
            # create a dictionnary (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device
            nb_of_batches = i + 1
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }
            self.get_G_loss(multi_domain_batch, verbose)

            if val_logger is None:
                val_logger = deepcopy(self.logger.losses.generator)
            else:
                val_logger = sum_dict(val_logger, self.logger.losses.generator)

        val_logger = div_dict(val_logger, nb_of_batches)
        self.logger.losses.generator = val_logger
        self.logger.log_losses(model_to_update="G", mode="val")

        for d in self.opts.domains:
            self.logger.log_comet_images("train", d)
            self.logger.log_comet_images("val", d)

        if "m" in self.opts.tasks and "p" in self.opts.tasks:
            self.logger.log_comet_combined_images("train", "r")
            self.logger.log_comet_combined_images("val", "r")

        if "m" in self.opts.tasks:
            self.eval_images("val", "r")
            self.eval_images("val", "s")

        if "p" in self.opts.tasks:
            val_fid = compute_val_fid(self)
            if self.exp is not None:
                self.exp.log_metric("val_fid", val_fid, step=self.logger.global_step)
            else:
                print("Validation FID Score", val_fid)

        self.train_mode()
        timing = int(time() - start_time)
        print("****************** Done in {}s *********************".format(timing))

    def eval_images(self, mode, domain):
        metrics = {"accuracy": accuracy, "mIOU": mIOU}
        metric_avg_scores = {"m": {}}
        if "s" in self.opts.tasks:
            metric_avg_scores["s"] = {}
        for key in metrics.keys():
            for task in metric_avg_scores.keys():
                metric_avg_scores[task][key] = 0.0

        if domain != "rf":
            for im_set in self.display_images[mode][domain]:
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)
                m = im_set["data"]["m"].unsqueeze(0).detach()
                z = self.G.encode(x)
                pred_mask = self.mask(z).detach().cpu()
                # Binarize mask
                pred_mask = (pred_mask > 0.5).to(torch.float32)
                for metric_key in metrics.keys():
                    metric_score = metrics[metric_key](pred_mask, m)
                    metric_avg_scores["m"][metric_key] += metric_score / len(
                        self.display_images[mode][domain]
                    )

                if "s" in self.opts.tasks:
                    pred_seg = self.G.decoders["s"](z).detach().cpu()
                    s = im_set["data"]["s"].unsqueeze(0).detach()
                    for metric_key in metrics.keys():
                        metric_score = metrics[metric_key](pred_seg, s)
                        metric_avg_scores["s"][metric_key] += metric_score / len(
                            self.display_images[mode][domain]
                        )

            if self.exp is not None:
                self.exp.log_metrics(
                    flatten_opts(metric_avg_scores),
                    prefix=f"metrics_{mode}_{domain}",
                    step=self.logger.global_step,
                )

        return 0

    def functional_test_mode(self):
        import atexit

        with open(Path(self.opts.output_path) / "is_functional.test", "w") as f:
            f.write("trainer functional test - delete this dir")

        if self.exp is not None:
            self.exp.log_parameter("is_functional_test", True)
        atexit.register(self.del_output_path)

    def del_output_path(self, force=False):
        import shutil

        if not Path(self.opts.output_path).exists():
            return

        if (Path(self.opts.output_path) / "is_functional.test").exists() or force:
            shutil.rmtree(self.opts.output_path)
