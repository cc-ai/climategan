"""
Main component: the trainer handles everything:
    * initializations
    * training
    * saving
"""
import inspect
import warnings
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
from comet_ml import ExistingExperiment, Experiment

warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
from addict import Dict
from torch import autograd, sigmoid, softmax
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from climategan.data import get_all_loaders
from climategan.discriminator import OmniDiscriminator, create_discriminator
from climategan.eval_metrics import accuracy, mIOU
from climategan.fid import compute_val_fid
from climategan.fire import add_fire
from climategan.generator import OmniGenerator, create_generator
from climategan.logger import Logger
from climategan.losses import get_losses
from climategan.optim import get_optimizer
from climategan.transforms import DiffTransforms
from climategan.tutils import (
    divide_pred,
    get_num_params,
    get_WGAN_gradient,
    lrgb2srgb,
    normalize,
    print_num_parameters,
    shuffle_batch_tuple,
    srgb2lrgb,
    vgg_preprocess,
    zero_grad,
)
from climategan.utils import (
    comet_kwargs,
    div_dict,
    find_target_size,
    flatten_opts,
    get_display_indices,
    get_existing_comet_id,
    get_latest_opts,
    merge,
    resolve,
    sum_dict,
    Timer,
)

try:
    import torch_xla.core.xla_model as xm  # type: ignore
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

        self.losses = None
        self.G = self.D = None
        self.real_val_fid_stats = None
        self.use_pl4m = False
        self.is_setup = False
        self.loaders = self.all_loaders = None
        self.exp = None

        self.current_mode = "train"
        self.diff_transforms = None
        self.kitti_pretrain = self.opts.train.kitti.pretrain
        self.pseudo_training_tasks = set(self.opts.train.pseudo.tasks)

        self.lr_names = {}
        self.base_display_images = {}
        self.kitty_display_images = {}
        self.domain_labels = {"s": 0, "r": 1}

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if isinstance(comet_exp, Experiment):
            self.exp = comet_exp

        if self.opts.train.amp:
            optimizers = [
                self.opts.gen.opt.optimizer.lower(),
                self.opts.dis.opt.optimizer.lower(),
            ]
            if "extraadam" in optimizers:
                raise ValueError(
                    "AMP does not work with ExtraAdam ({})".format(optimizers)
                )
            self.grad_scaler_d = GradScaler()
            self.grad_scaler_g = GradScaler()

        # -------------------------------
        # -----  Legacy Overwrites  -----
        # -------------------------------
        if (
            self.opts.gen.s.depth_feat_fusion is True
            or self.opts.gen.s.depth_dada_fusion is True
        ):
            self.opts.gen.s.use_dada = True

    @torch.no_grad()
    def paint_and_mask(self, image_batch, mask_batch=None, resolution="approx"):
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
            mask_batch = self.G.mask(x=image_batch)
        else:
            assert len(image_batch) == len(mask_batch)
            assert image_batch.shape[-2:] == mask_batch.shape[-2:]

        if resolution not in {"approx", "exact"}:
            painted = self.G.paint(mask_batch, image_batch)

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
                image_batch.shape[-2] // 2**self.opts.gen.p.spade_n_up
            )
            self.G.painter.z_w = (
                image_batch.shape[-1] // 2**self.opts.gen.p.spade_n_up
            )

            painted = self.G.paint(mask_batch, image_batch)

            self.G.painter.z_h = zh
            self.G.painter.z_w = zw
            if resolution == "exact":
                painted = nn.functional.interpolate(
                    painted, size=image_batch.shape[-2:], mode="bilinear"
                )

        if previous_mode == "train":
            self.train_mode()

        return painted

    def _p(self, *args, **kwargs):
        """
        verbose-dependant print util
        """
        if self.verbose > 0:
            print(*args, **kwargs)

    @torch.no_grad()
    def infer_all(
        self,
        x,
        numpy=True,
        stores={},
        bin_value=-1,
        half=False,
        xla=False,
        cloudy=False,
        auto_resize_640=False,
        ignore_event=set(),
        return_masks=False,
    ):
        """
        Create a dictionnary of events from a numpy or tensor,
        single or batch image data.

        stores is a dictionnary of times for the Timer class.

        bin_value is used to binarize (or not) flood masks
        """
        assert self.is_setup
        assert len(x.shape) in {3, 4}, f"Unknown Data shape {x.shape}"

        # convert numpy to tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        # add batch dimension
        if len(x.shape) == 3:
            x.unsqueeze_(0)

        # permute channels as second dimension
        if x.shape[1] != 3:
            assert x.shape[-1] == 3, f"Unknown x shape to permute {x.shape}"
            x = x.permute(0, 3, 1, 2)

        # send to device
        if x.device != self.device:
            x = x.to(self.device)

        # interpolate to standard input size
        if auto_resize_640 and (x.shape[-1] != 640 or x.shape[-2] != 640):
            x = torch.nn.functional.interpolate(x, (640, 640), mode="bilinear")

        if half:
            x = x.half()

        # adjust painter's latent vector
        self.G.painter.set_latent_shape(x.shape, True)

        with Timer(store=stores.get("all events", [])):
            # encode
            with Timer(store=stores.get("encode", [])):
                z = self.G.encode(x)
                if xla:
                    xm.mark_step()

            # predict from masker
            with Timer(store=stores.get("depth", [])):
                depth, z_depth = self.G.decoders["d"](z)
                if xla:
                    xm.mark_step()
            with Timer(store=stores.get("segmentation", [])):
                segmentation = self.G.decoders["s"](z, z_depth)
                if xla:
                    xm.mark_step()
            with Timer(store=stores.get("mask", [])):
                cond = self.G.make_m_cond(depth, segmentation, x)
                mask = self.G.mask(z=z, cond=cond, z_depth=z_depth)
                if xla:
                    xm.mark_step()

            # apply events
            if "wildfire" not in ignore_event:
                with Timer(store=stores.get("wildfire", [])):
                    wildfire = self.compute_fire(x, seg_preds=segmentation)
            if "smog" not in ignore_event:
                with Timer(store=stores.get("smog", [])):
                    smog = self.compute_smog(x, d=depth, s=segmentation)
            if "flood" not in ignore_event:
                with Timer(store=stores.get("flood", [])):
                    flood = self.compute_flood(
                        x,
                        m=mask,
                        s=segmentation,
                        cloudy=cloudy,
                        bin_value=bin_value,
                    )

        if xla:
            xm.mark_step()

        if numpy:
            with Timer(store=stores.get("numpy", [])):
                # normalize to 0-1
                flood = normalize(flood).cpu()
                smog = normalize(smog).cpu()
                wildfire = normalize(wildfire).cpu()

                # convert to numpy
                flood = flood.permute(0, 2, 3, 1).numpy()
                smog = smog.permute(0, 2, 3, 1).numpy()
                wildfire = wildfire.permute(0, 2, 3, 1).numpy()

                # convert to 0-255 uint8
                flood = (flood * 255).astype(np.uint8)
                smog = (smog * 255).astype(np.uint8)
                wildfire = (wildfire * 255).astype(np.uint8)

        output_data = {"flood": flood, "wildfire": wildfire, "smog": smog}
        if return_masks:
            output_data["mask"] = (
                ((mask > bin_value) * 255).cpu().numpy().astype(np.uint8)
            )

        return output_data

    @classmethod
    def resume_from_path(
        cls,
        path,
        overrides={},
        setup=True,
        inference=False,
        new_exp=False,
        device=None,
        verbose=1,
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
            device (torch.device, optional): Device to use

        Returns:
            climategan.Trainer: Loaded and resumed trainer
        """
        p = resolve(path)
        assert p.exists()

        c = p / "checkpoints"
        assert c.exists() and c.is_dir()

        opts = get_latest_opts(p)
        opts = Dict(merge(overrides, opts))
        opts.train.resume = True

        if new_exp is None:
            exp = None
        elif new_exp is True:
            exp = Experiment(project_name="climategan", **comet_kwargs)
            exp.log_asset_folder(
                str(resolve(Path(__file__)).parent),
                recursive=True,
                log_file_name=True,
            )
            exp.log_parameters(flatten_opts(opts))
        else:
            comet_id = get_existing_comet_id(p)
            exp = ExistingExperiment(previous_experiment=comet_id, **comet_kwargs)

        trainer = cls(opts, comet_exp=exp, device=device, verbose=verbose)

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

                print(f"Resuming P+M model from \n  -{p_path} \nand \n  -{m_path}")
                m_checkpoint = torch.load(m_path, map_location=map_loc)
                p_checkpoint = torch.load(p_path, map_location=map_loc)
                checkpoint = merge(m_checkpoint, p_checkpoint)

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

            print(f"Resuming {model} model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=map_loc)

        # On TPUs must send the data to the xla device as it cannot be mapped
        # there directly from torch.load
        if tpu:
            checkpoint = xm.send_cpu_data_to_device(checkpoint, self.device)

        # -----------------------
        # -----  Restore G  -----
        # -----------------------
        if inference:
            incompatible_keys = self.G.load_state_dict(checkpoint["G"], strict=False)
            if incompatible_keys.missing_keys:
                print("WARNING: Missing keys in self.G.load_state_dict, keeping inits")
                print(incompatible_keys.missing_keys)
            if incompatible_keys.unexpected_keys:
                print("WARNING: Ignoring Unexpected keys in self.G.load_state_dict")
                print(incompatible_keys.unexpected_keys)
        else:
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

        # -----------------------
        # -----  Restore D  -----
        # -----------------------
        if self.D is not None and get_num_params(self.D) > 0:
            self.D.load_state_dict(checkpoint["D"])
            self.d_opt.load_state_dict(checkpoint["d_opt"])

        # ---------------------------
        # -----  Resore logger  -----
        # ---------------------------
        self.logger.epoch = checkpoint["epoch"]
        self.logger.global_step = checkpoint["step"]
        self.exp.log_text(
            "Resuming from epoch {} & step {}".format(
                checkpoint["epoch"], checkpoint["step"]
            )
        )
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
        self.current_mode = "eval"

    def train_mode(self):
        """
        Set trainer's models in train mode
        """
        if self.G is not None:
            self.G.train()
        if self.D is not None:
            self.D.train()

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
        return self.G.sample_painter_z(batch_size, self.device)

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
        for mode in self.all_loaders:
            for domain in self.all_loaders.loaders[mode]:
                x = (
                    self.all_loaders[mode][domain]
                    .dataset[0]["data"]["x"]
                    .to(self.device)
                )
                break
            if x is not None:
                break

        if x is None:
            raise ValueError("No batch found to compute_latent_shape")

        x = x.unsqueeze(0)
        z = self.G.encode(x)
        return z.shape[1:] if not isinstance(z, (list, tuple)) else z[0].shape[1:]

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

    def update_learning_rates(self):
        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()

    def setup(self, inference=False):
        """Prepare the trainer before it can be used to train the models:
        * initialize G and D
        * creates 2 optimizers
        """
        self.logger.global_step = 0
        start_time = time()
        self.logger.time.start_time = start_time
        verbose = self.verbose

        if not inference:
            self.all_loaders = get_all_loaders(self.opts)

        # -----------------------
        # -----  Generator  -----
        # -----------------------
        __t = time()
        print("Creating generator...")

        self.G: OmniGenerator = create_generator(
            self.opts, device=self.device, no_init=inference, verbose=verbose
        )

        self.has_painter = get_num_params(self.G.painter) or self.G.load_val_painter()

        if self.has_painter:
            self.G.painter.set_latent_shape(find_target_size(self.opts, "x"), True)

        print(f"Generator OK in {time() - __t:.1f}s.")

        if inference:  # Inference mode: no more than a Generator needed
            print("Inference mode: no Discriminator, no optimizers")
            print_num_parameters(self)
            self.switch_data(to="base")
            if self.opts.train.resume:
                self.resume(True)
            self.eval_mode()
            print("Trainer is in evaluation mode.")
            print("Setup done.")
            self.is_setup = True
            return

        # ---------------------------
        # -----  Discriminator  -----
        # ---------------------------

        self.D: OmniDiscriminator = create_discriminator(
            self.opts, self.device, verbose=verbose
        )
        print("Discriminator OK.")

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
                self.D, self.opts.dis.opt, self.opts.tasks, True
            )
        else:
            self.d_opt, self.d_scheduler = None, None

        self.losses = get_losses(self.opts, verbose, device=self.device)

        if "p" in self.opts.tasks and self.opts.gen.p.diff_aug.use:
            self.diff_transforms = DiffTransforms(self.opts.gen.p.diff_aug)

        if verbose > 0:
            for mode, mode_dict in self.all_loaders.items():
                for domain, domain_loader in mode_dict.items():
                    print(
                        "Loader {} {} : {}".format(
                            mode, domain, len(domain_loader.dataset)
                        )
                    )

        # ----------------------------
        # -----  Display images  -----
        # ----------------------------
        self.set_display_images()

        # -------------------------------
        # -----  Log Architectures  -----
        # -------------------------------
        self.logger.log_architecture()

        # -----------------------------
        # -----  Set data source  -----
        # -----------------------------
        if self.kitti_pretrain:
            self.switch_data(to="kitti")
        else:
            self.switch_data(to="base")

        # -------------------------
        # -----  Setup Done.  -----
        # -------------------------
        print(" " * 50, end="\r")
        print("Done creating display images")

        if self.opts.train.resume:
            print("Resuming Model (inference: False)")
            self.resume(False)
        else:
            print("Not resuming: starting a new model")

        print("Setup done.")
        self.is_setup = True

    def switch_data(self, to="kitti"):
        caller = inspect.stack()[1].function
        print(f"[{caller}] Switching data source to", to)
        self.data_source = to
        if to == "kitti":
            self.display_images = self.kitty_display_images
            if self.all_loaders is not None:
                self.loaders = {
                    mode: {"s": self.all_loaders[mode]["kitti"]}
                    for mode in self.all_loaders
                }
        else:
            self.display_images = self.base_display_images
            if self.all_loaders is not None:
                self.loaders = {
                    mode: {
                        domain: self.all_loaders[mode][domain]
                        for domain in self.all_loaders[mode]
                        if domain != "kitti"
                    }
                    for mode in self.all_loaders
                }
        if (
            self.logger.global_step % 2 != 0
            and "extra" in self.opts.dis.opt.optimizer.lower()
        ):
            print(
                "Warning: artificially bumping step to run an extrapolation step first."
            )
            self.logger.global_step += 1

    def set_display_images(self, use_all=False):
        for mode, mode_dict in self.all_loaders.items():

            if self.kitti_pretrain:
                self.kitty_display_images[mode] = {}
            self.base_display_images[mode] = {}

            for domain in mode_dict:

                if self.kitti_pretrain and domain == "kitti":
                    target_dict = self.kitty_display_images
                else:
                    if domain == "kitti":
                        continue
                    target_dict = self.base_display_images

                dataset = self.all_loaders[mode][domain].dataset
                display_indices = (
                    get_display_indices(self.opts, domain, len(dataset))
                    if not use_all
                    else list(range(len(dataset)))
                )
                ldis = len(display_indices)
                print(
                    f"       Creating {ldis} {mode} {domain} display images...",
                    end="\r",
                    flush=True,
                )
                target_dict[mode][domain] = [
                    Dict(dataset[i])
                    for i in display_indices
                    if (print(f"({i})", end="\r") is None and i < len(dataset))
                ]
                if self.exp is not None:
                    for im_id, d in enumerate(target_dict[mode][domain]):
                        self.exp.log_parameter(
                            "display_image_{}_{}_{}".format(mode, domain, im_id),
                            d["paths"],
                        )

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
            # backprop painter's disc loss to masker
            if (
                self.logger.epoch == self.opts.gen.p.pl4m_epoch
                and get_num_params(self.G.painter) > 0
                and "p" in self.opts.tasks
                and self.opts.gen.m.use_pl4m
            ):
                print(
                    "\n\n >>> Enabling pl4m at epoch {}\n\n".format(self.logger.epoch)
                )
                self.use_pl4m = True

            self.run_epoch()
            self.run_evaluation(verbose=1)
            self.save()

            # end vkitti2 pre-training
            if self.logger.epoch == self.opts.train.kitti.epochs - 1:
                self.switch_data(to="base")
                self.kitti_pretrain = False

            # end pseudo training
            if self.logger.epoch == self.opts.train.pseudo.epochs - 1:
                self.pseudo_training_tasks = set()

    def run_epoch(self):
        """Runs an epoch:
        * checks trainer is setup
        * gets a tuple of batches per domain
        * sends batches to device
        * updates sequentially G, D
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
            if self.d_opt is not None and not self.kitti_pretrain:
                for param in self.D.parameters():
                    param.requires_grad = True

                self.update_D(multi_domain_batch)

            # -------------------------
            # -----  Log Metrics  -----
            # -------------------------
            self.logger.global_step += 1
            self.logger.log_step_time(time())

        if not self.kitti_pretrain:
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
            if domain == "rf" and self.has_painter:
                m = batch["data"]["m"]
                # sample vector
                with torch.no_grad():
                    # see spade compute_discriminator_loss
                    fake = self.G.paint(m, x)
                    if self.opts.gen.p.diff_aug.use:
                        fake = self.diff_transforms(fake)
                        x = self.diff_transforms(x)
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
                s_pred = d_pred = cond = z_depth = None

                if "s" in batch["data"]:
                    if "d" in self.opts.tasks and self.opts.gen.s.use_dada:
                        d_pred, z_depth = self.G.decoders["d"](z)

                    step_loss, s_pred = self.masker_s_loss(
                        x, z, d_pred, z_depth, None, domain, for_="D"
                    )
                    step_loss *= self.opts.train.lambdas.advent.adv_main
                    disc_loss["s"]["Advent"] += step_loss

                if "m" in batch["data"]:
                    if "d" in self.opts.tasks:
                        if self.opts.gen.m.use_spade:
                            if d_pred is None:
                                d_pred, z_depth = self.G.decoders["d"](z)
                            cond = self.G.make_m_cond(d_pred, s_pred, x)
                        elif self.opts.gen.m.use_dada:
                            if d_pred is None:
                                d_pred, z_depth = self.G.decoders["d"](z)

                    step_loss, _ = self.masker_m_loss(
                        x,
                        z,
                        None,
                        domain,
                        for_="D",
                        cond=cond,
                        z_depth=z_depth,
                        depth_preds=d_pred,
                    )
                    step_loss *= self.opts.train.lambdas.advent.adv_main
                    disc_loss["m"]["Advent"] += step_loss

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

        if "p" in self.opts.tasks and not self.kitti_pretrain:
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
            * compute task-specific losses
            * compute the adaptation and translation decoders' auto-encoding losses
            * compute the adaptation decoder's translation losses (GAN and Cycle)

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

            # --------------------------------------
            # -----  task-specific losses (2)  -----
            # --------------------------------------
            d_pred = s_pred = z_depth = None
            for task in ["d", "s", "m"]:
                if task not in batch["data"]:
                    continue

                target = batch["data"][task]

                if task == "d":
                    loss, d_pred, z_depth = self.masker_d_loss(
                        x, z, target, domain, "G"
                    )
                    m_loss += loss
                    self.logger.losses.gen.task["d"][domain] = loss.item()

                elif task == "s":
                    loss, s_pred = self.masker_s_loss(
                        x, z, d_pred, z_depth, target, domain, "G"
                    )
                    m_loss += loss
                    self.logger.losses.gen.task["s"][domain] = loss.item()

                elif task == "m":
                    cond = None
                    if self.opts.gen.m.use_spade:
                        if not self.opts.gen.m.detach:
                            d_pred = d_pred.clone()
                            s_pred = s_pred.clone()
                        cond = self.G.make_m_cond(d_pred, s_pred, x)

                    loss, _ = self.masker_m_loss(
                        x,
                        z,
                        target,
                        domain,
                        "G",
                        cond=cond,
                        z_depth=z_depth,
                        depth_preds=d_pred,
                    )
                    m_loss += loss
                    self.logger.losses.gen.task["m"][domain] = loss.item()

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
        fake_flooded = self.G.paint(m, x)

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
        if self.opts.gen.p.diff_aug.use:
            fake_flooded = self.diff_transforms(fake_flooded)
            x = self.diff_transforms(x)

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

    def masker_d_loss(self, x, z, target, domain, for_="G"):
        assert for_ in {"G", "D"}
        self.assert_z_matches_x(x, z)
        assert x.shape[0] == target.shape[0]
        zero_loss = torch.tensor(0.0, device=self.device)
        weight = self.opts.train.lambdas.G.d.main

        prediction, z_depth = self.G.decoders["d"](z)

        if self.opts.gen.d.classify.enable:
            target.squeeze_(1)

        full_loss = self.losses["G"]["tasks"]["d"](prediction, target)
        full_loss *= weight

        if weight == 0 or (domain == "r" and "d" not in self.pseudo_training_tasks):
            return zero_loss, prediction, z_depth

        return full_loss, prediction, z_depth

    def masker_s_loss(self, x, z, depth_preds, z_depth, target, domain, for_="G"):
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
            pred = self.G.decoders["s"](z, z_depth)

        # Supervised segmentation loss: crossent for sim domain,
        # crossent_pseudo for real ; loss is crossent in any case
        if for_ == "G":
            if domain == "s" or "s" in self.pseudo_training_tasks:
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
            if self.opts.gen.s.use_dada and depth_preds is not None:
                depth_preds = depth_preds.detach()
            else:
                depth_preds = None

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
                    depth_preds,
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

        return full_loss, pred

    def masker_m_loss(
        self, x, z, target, domain, for_="G", cond=None, z_depth=None, depth_preds=None
    ):
        assert for_ in {"G", "D"}
        assert domain in {"r", "s"}
        self.assert_z_matches_x(x, z)
        assert x.shape[0] == target.shape[0] if target is not None else True
        full_loss = torch.tensor(0.0, device=self.device)

        pred_logits = self.G.decoders["m"](z, cond=cond, z_depth=z_depth)
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
            if self.opts.gen.m.use_dada and depth_preds is not None:
                depth_preds = depth_preds.detach()
                depth_preds = torch.nn.functional.interpolate(
                    depth_preds, size=x.shape[-2:], mode="nearest"
                )
            else:
                depth_preds = None

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
                    depth_preds,
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

        return full_loss, prob

    def painter_loss_for_masker(self, x, m):
        # pl4m loss
        # painter should not be updated
        for param in self.G.painter.parameters():
            param.requires_grad = False
        # TODO for param in self.D.painter.parameters():
        #     param.requires_grad = False

        fake_flooded = self.G.paint(m, x)

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

        if "m" in self.opts.tasks and self.has_painter and not self.kitti_pretrain:
            self.logger.log_comet_combined_images("train", "r")
            self.logger.log_comet_combined_images("val", "r")

        if self.exp is not None:
            print()

        if "m" in self.opts.tasks or "s" in self.opts.tasks:
            self.eval_images("val", "r")
            self.eval_images("val", "s")

        if "p" in self.opts.tasks and not self.kitti_pretrain:
            val_fid = compute_val_fid(self)
            if self.exp is not None:
                self.exp.log_metric("val_fid", val_fid, step=self.logger.global_step)
            else:
                print("Validation FID Score", val_fid)

        self.train_mode()
        timing = int(time() - start_time)
        print("****************** Done in {}s *********************".format(timing))

    def eval_images(self, mode, domain):
        if domain == "s" and self.kitti_pretrain:
            domain = "kitti"
        if domain == "rf" or domain not in self.display_images[mode]:
            return

        metric_funcs = {"accuracy": accuracy, "mIOU": mIOU}
        metric_avg_scores = {"m": {}}
        if "s" in self.opts.tasks:
            metric_avg_scores["s"] = {}
        if "d" in self.opts.tasks and domain == "s" and self.opts.gen.d.classify.enable:
            metric_avg_scores["d"] = {}

        for key in metric_funcs:
            for task in metric_avg_scores:
                metric_avg_scores[task][key] = []

        for im_set in self.display_images[mode][domain]:
            x = im_set["data"]["x"].unsqueeze(0).to(self.device)
            z = self.G.encode(x)

            s_pred = d_pred = z_depth = None

            if "d" in metric_avg_scores:
                d_pred, z_depth = self.G.decoders["d"](z)
                d_pred = d_pred.detach().cpu()

                if domain == "s":
                    d = im_set["data"]["d"].unsqueeze(0).detach()

                    for metric in metric_funcs:
                        metric_score = metric_funcs[metric](d_pred, d)
                        metric_avg_scores["d"][metric].append(metric_score)

            if "s" in metric_avg_scores:
                if z_depth is None:
                    if self.opts.gen.s.use_dada and "d" in self.opts.tasks:
                        _, z_depth = self.G.decoders["d"](z)
                s_pred = self.G.decoders["s"](z, z_depth).detach().cpu()
                s = im_set["data"]["s"].unsqueeze(0).detach()

                for metric in metric_funcs:
                    metric_score = metric_funcs[metric](s_pred, s)
                    metric_avg_scores["s"][metric].append(metric_score)

            if "m" in self.opts:
                cond = None
                if s_pred is not None and d_pred is not None:
                    cond = self.G.make_m_cond(d_pred, s_pred, x)
                if z_depth is None:
                    if self.opts.gen.m.use_dada and "d" in self.opts.tasks:
                        _, z_depth = self.G.decoders["d"](z)

                pred_mask = (
                    (self.G.mask(z=z, cond=cond, z_depth=z_depth)).detach().cpu()
                )
                pred_mask = (pred_mask > 0.5).to(torch.float32)
                pred_prob = torch.cat([1 - pred_mask, pred_mask], dim=1)

                m = im_set["data"]["m"].unsqueeze(0).detach()

                for metric in metric_funcs:
                    if metric != "mIOU":
                        metric_score = metric_funcs[metric](pred_mask, m)
                    else:
                        metric_score = metric_funcs[metric](pred_prob, m)

                    metric_avg_scores["m"][metric].append(metric_score)

        metric_avg_scores = {
            task: {
                metric: np.mean(values) if values else float("nan")
                for metric, values in met_dict.items()
            }
            for task, met_dict in metric_avg_scores.items()
        }
        metric_avg_scores = {
            task: {
                metric: value if not np.isnan(value) else -1
                for metric, value in met_dict.items()
            }
            for task, met_dict in metric_avg_scores.items()
        }
        if self.exp is not None:
            self.exp.log_metrics(
                flatten_opts(metric_avg_scores),
                prefix=f"metrics_{mode}_{domain}",
                step=self.logger.global_step,
            )
        else:
            print(f"metrics_{mode}_{domain}")
            print(flatten_opts(metric_avg_scores))

        return 0

    def functional_test_mode(self):
        import atexit

        self.opts.output_path = (
            Path("~").expanduser() / "climategan" / "functional_tests"
        )
        Path(self.opts.output_path).mkdir(parents=True, exist_ok=True)
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

    def compute_fire(self, x, seg_preds=None, z=None, z_depth=None):
        """
        Transforms input tensor given wildfires event
        Args:
            x (torch.Tensor): Input tensor
                seg_preds (torch.Tensor): Semantic segmentation
                predictions for input tensor
            z (torch.Tensor): Latent vector of encoded "x".
                Can be None if seg_preds is given.
        Returns:
            torch.Tensor: Wildfire version of input tensor
        """

        if seg_preds is None:
            if z is None:
                z = self.G.encode(x)
            seg_preds = self.G.decoders["s"](z, z_depth)

        return add_fire(x, seg_preds, self.opts.events.fire)

    def compute_flood(
        self, x, z=None, z_depth=None, m=None, s=None, cloudy=None, bin_value=-1
    ):
        """
        Applies a flood (mask + paint) to an input image, with optionally
        pre-computed masker z or mask

        Args:
            x (torch.Tensor): B x C x H x W -1:1 input image
            z (torch.Tensor, optional): B x C x H x W Masker latent vector.
                Defaults to None.
            m (torch.Tensor, optional): B x 1 x H x W Mask. Defaults to None.
            bin_value (float, optional): Mask binarization value.
                Set to -1 to use smooth masks (no binarization)

        Returns:
            torch.Tensor: B x 3 x H x W -1:1 flooded image
        """

        if m is None:
            if z is None:
                z = self.G.encode(x)
            if "d" in self.opts.tasks and self.opts.gen.m.use_dada and z_depth is None:
                _, z_depth = self.G.decoders["d"](z)
            m = self.G.mask(x=x, z=z, z_depth=z_depth)

        if bin_value >= 0:
            m = (m > bin_value).to(m.dtype)

        if cloudy:
            assert s is not None
            return self.G.paint_cloudy(m, x, s)

        return self.G.paint(m, x)

    def compute_smog(self, x, z=None, d=None, s=None, use_sky_seg=False):
        # implementation from the paper:
        # HazeRD: An outdoor scene dataset and benchmark for single image dehazing
        sky_mask = None
        if d is None or (use_sky_seg and s is None):
            if z is None:
                z = self.G.encode(x)
            if d is None:
                d, _ = self.G.decoders["d"](z)
            if use_sky_seg and s is None:
                if "s" not in self.opts.tasks:
                    raise ValueError(
                        "Cannot have "
                        + "(use_sky_seg is True and s is None and 's' not in tasks)"
                    )
                s = self.G.decoders["s"](z)
                # TODO: s to sky mask
                # TODO: interpolate to d's size

        params = self.opts.events.smog

        airlight = params.airlight * torch.ones(3)
        airlight = airlight.view(1, -1, 1, 1).to(self.device)

        irradiance = srgb2lrgb(x)

        beta = torch.tensor([params.beta / params.vr] * 3)
        beta = beta.view(1, -1, 1, 1).to(self.device)

        d = normalize(d, mini=0.3, maxi=1.0)
        d = 1.0 / d
        d = normalize(d, mini=0.1, maxi=1)

        if sky_mask is not None:
            d[sky_mask] = 1

        d = torch.nn.functional.interpolate(
            d, size=x.shape[-2:], mode="bilinear", align_corners=True
        )

        d = d.repeat(1, 3, 1, 1)

        transmission = torch.exp(d * -beta)

        smogged = transmission * irradiance + (1 - transmission) * airlight

        smogged = lrgb2srgb(smogged)

        # add yellow filter
        alpha = params.alpha / 255
        yellow_mask = torch.Tensor([params.yellow_color]) / 255
        yellow_filter = (
            yellow_mask.unsqueeze(2)
            .unsqueeze(2)
            .repeat(1, 1, smogged.shape[-2], smogged.shape[-1])
            .to(self.device)
        )

        smogged = smogged * (1 - alpha) + yellow_filter * alpha

        return smogged
