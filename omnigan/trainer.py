"""
Main component: the trainer handles everything:
    * initializations
    * training
    * saving
"""
import os
import warnings
from copy import deepcopy
from pathlib import Path
from time import time

from comet_ml import ExistingExperiment

warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import torchvision.utils as vutils
from addict import Dict
from comet_ml import Experiment
from torch import autograd
from tqdm import tqdm

from omnigan.classifier import OmniClassifier, get_classifier
from omnigan.data import decode_segmap_merged_labels, get_all_loaders
from omnigan.discriminator import OmniDiscriminator, get_dis
from omnigan.eval_metrics import accuracy, iou
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
    norm_tensor,
    shuffle_batch_tuple,
    vgg_preprocess,
    zero_grad,
)
from omnigan.utils import (
    comet_kwargs,
    div_dict,
    flatten_opts,
    get_display_indices,
    get_existing_comet_id,
    get_latest_path,
    get_latest_opts,
    merge,
    sum_dict,
)

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
        self.logger = Dict()
        self.logger.epoch = 0
        self.loaders = None
        self.losses = None
        self.input_shape = None
        self.G = self.D = self.C = None
        self.lr_names = {}
        self.no_z = self.opts.gen.p.no_z
        self.real_val_fid_stats = None

        self.is_setup = False
        self.current_mode = "train"

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.exp = None
        if isinstance(comet_exp, Experiment):
            self.exp = comet_exp
        self.domain_labels = {"s": 0, "r": 1}

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
            mask_batch = self.G.decoders["m"](z)
        else:
            assert len(image_batch) == len(mask_batch)
            assert image_batch.shape[-2:] == mask_batch.shape[-2:]

        z_painter = None
        masked_batch = image_batch * (1.0 - mask_batch)

        if resolution not in {"approx", "exact"}:
            painted = self.G.painter(z_painter, masked_batch)
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

    def log_losses(self, model_to_update="G", mode="train"):
        """Logs metrics on comet.ml

        Args:
            model_to_update (str, optional): One of "G", "D" or "C". Defaults to "G".
        """
        loss_names = {"G": "gen", "D": "disc", "C": "classifier"}

        if self.opts.train.log_level < 1:
            return

        if self.exp is None:
            return

        assert model_to_update in {
            "G",
            "D",
            "C",
        }, "unknown model to log losses {}".format(model_to_update)

        loss_to_update = self.logger.losses[loss_names[model_to_update]]

        losses = loss_to_update.copy()

        if self.opts.train.log_level == 1:
            # Only log aggregated losses: delete other keys in losses
            for k in loss_to_update:
                if k not in {"masker", "total_loss", "painter"}:
                    del losses[k]
        # convert losses into a single-level dictionnary

        losses = flatten_opts(losses)
        self.exp.log_metrics(
            losses, prefix=f"{model_to_update}_{mode}", step=self.logger.global_step
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
        return z.shape[1:]

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

    def print_num_parameters(self):
        print("---------------------------")
        if self.G.encoder is not None:
            print("num params encoder: ", get_num_params(self.G.encoder))
        for d in self.G.decoders.keys():
            print(
                "num params decoder {}: {}".format(
                    d, get_num_params(self.G.decoders[d])
                )
            )
        print("num params painter: ", get_num_params(self.G.painter))

        if self.D is not None:
            for d in self.D.keys():
                print("num params discrim {}: {}".format(d, get_num_params(self.D[d])))

        if self.C is not None:
            print("num params classif: ", get_num_params(self.C))
        print("---------------------------")

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
        print(f"Generator OK in {time() - __t:.1f}s.", end="", flush=True)

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
            self.print_num_parameters()
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

        self.print_num_parameters()

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
        print("Creating display images...", end="", flush=True)
        self.display_images = {}
        for mode, mode_dict in self.loaders.items():
            self.display_images[mode] = {}
            for domain, domain_loader in mode_dict.items():
                dataset = self.loaders[mode][domain].dataset
                display_indices = get_display_indices(self.opts, domain, len(dataset))
                self.display_images[mode][domain] = [
                    Dict(dataset[i]) for i in display_indices if i < len(dataset)
                ]
                if self.exp is not None:
                    for im_id, d in enumerate(self.display_images[mode][domain]):
                        self.exp.log_parameter(
                            "display_image_{}_{}_{}".format(mode, domain, im_id),
                            d["paths"],
                        )

        print("Setup done.")
        self.is_setup = True

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

    @property
    def train_loaders(self):
        """Get a zip of all training loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["train"].values()))

    def update_learning_rates(self):
        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()
        if self.c_scheduler is not None:
            self.c_scheduler.step()

    def log_learning_rates(self):
        lrs = {}
        if self.g_scheduler is not None:
            for name, lr in zip(self.lr_names["G"], self.g_scheduler.get_last_lr()):
                lrs[f"lr_G_{name}"] = lr
        if self.d_scheduler is not None:
            for name, lr in zip(self.lr_names["D"], self.d_scheduler.get_last_lr()):
                lrs[f"lr_D_{name}"] = lr
        if self.c_scheduler is not None:
            for name, lr in zip(self.lr_names["C"], self.c_scheduler.get_last_lr()):
                lrs[f"lr_C_{name}"] = lr
        self.exp.log_metrics(lrs, step=self.logger.global_step)

    @property
    def val_loaders(self):
        """Get a zip of all validation loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["val"].values()))

    def run_epoch(self):
        """Runs an epoch:
        * checks trainer is setup
        * gets a tuple of batches per domain
        * sends batches to device
        * updates sequentially G, D, C
        """
        assert self.is_setup
        self.train_mode()
        self.exp.log_parameter("epoch", self.logger.epoch)
        epoch_len = min(len(loader) for loader in self.loaders["train"].values())
        epoch_desc = "Epoch {}".format(self.logger.epoch)
        for multi_batch_tuple in tqdm(
            self.train_loaders,
            desc=epoch_desc,
            total=epoch_len,
            mininterval=0.5,
            unit="batch",
        ):
            # create a dictionnay (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device

            step_start_time = time()
            multi_batch_tuple = shuffle_batch_tuple(multi_batch_tuple)

            # The `[0]` is because the domain is contained in a list
            # i.e. domain "r" is ["r"]
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }

            if self.d_opt is not None:
                # freeze params of the discriminator
                for param in self.D.parameters():
                    param.requires_grad = False

            # ------------------------------
            # -----  Update Generator  -----
            # ------------------------------
            self.update_g(multi_domain_batch)

            # ----------------------------------
            # -----  Update Discriminator  -----
            # ----------------------------------
            if self.d_opt is not None:
                # unfreeze params of advent discriminator
                for param in self.D.parameters():
                    param.requires_grad = True

                self.update_d(multi_domain_batch)

            # -------------------------------
            # -----  Update Classifier  -----
            # -------------------------------
            if self.opts.train.latent_domain_adaptation and self.C is not None:
                self.update_c(multi_domain_batch)

            # -----------------
            # -----  Log  -----
            # -----------------
            self.logger.global_step += 1
            step_time = time() - step_start_time
            self.log_step_time(step_time)

        self.update_learning_rates()
        self.log_learning_rates()

    def log_step_time(self, step_time):
        """Logs step-time on comet.ml

        Args:
            step_time (float): step-time in seconds
        """
        if self.exp:
            self.exp.log_metric("Step-time", step_time, step=self.logger.global_step)

    def log_comet_images(self, mode, domain):
        save_images = {}
        if domain != "rf":
            for j, im_set in enumerate(self.display_images[mode][domain]):
                print(j, end="\r")
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)
                self.z = self.G.encode(x)

                if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                    depth_prediction, z_depth = self.G.decoders["d"](self.z)
                    z_feat_fusion = self.z * z_depth

                for update_task, update_target in im_set["data"].items():
                    target = im_set["data"][update_task].unsqueeze(0).to(self.device)
                    task_saves = []

                    if update_task == "x":
                        continue

                    if update_task not in save_images:
                        save_images[update_task] = []

                    if update_task == "s":
                        if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                            prediction = self.G.decoders[update_task](z_feat_fusion)
                        else:
                            prediction = self.G.decoders[update_task](self.z)
                        target = (
                            decode_segmap_merged_labels(target, domain, True)
                            .float()
                            .to(self.device)
                        )
                        prediction = (
                            decode_segmap_merged_labels(prediction, domain, False)
                            .float()
                            .to(self.device)
                        )
                        task_saves.append(target)

                    elif update_task == "m":
                        if "d" in self.opts.tasks and self.opts.gen.m.do_feat_fusion:
                            prediction = self.G.decoders[update_task](z_feat_fusion)
                        else:
                            prediction = self.G.decoders[update_task](self.z)
                        prediction = prediction.repeat(1, 3, 1, 1)
                        task_saves.append(x * (1.0 - prediction))
                        task_saves.append(x * (1.0 - target.repeat(1, 3, 1, 1)))

                    elif update_task == "d":
                        if self.opts.gen.d.use_dada:
                            prediction = depth_prediction
                        else:
                            prediction = self.G.decoders[update_task](self.z)
                        # prediction is a log depth tensor
                        target = (norm_tensor(target)) * 255
                        prediction = (norm_tensor(prediction)) * 255
                        prediction = prediction.repeat(1, 3, 1, 1)
                        task_saves.append(target.repeat(1, 3, 1, 1))

                    task_saves.append(prediction)
                    save_images[update_task].append(x.cpu().detach())

                    for im in task_saves:
                        save_images[update_task].append(im.cpu().detach())

            for task in save_images.keys():
                # Write images:
                self.write_images(
                    image_outputs=save_images[task],
                    mode=mode,
                    domain=domain,
                    task=task,
                    im_per_row=self.opts.comet.im_per_row.get(task, 4),
                    rows_per_log=self.opts.comet.get("rows_per_log", 5),
                    comet_exp=self.exp,
                )
        else:
            # in the rf domain display_size may be different from fid.n_images
            limit = self.opts.comet.display_size
            image_outputs = []
            for im_set in self.display_images[mode][domain][:limit]:
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)
                m = im_set["data"]["m"].unsqueeze(0).to(self.device)

                z = self.sample_z(x.shape[0]) if not self.no_z else None
                prediction = self.G.painter(z, x * (1.0 - m))
                image_outputs.append(x * (1.0 - m))
                image_outputs.append(prediction)
                image_outputs.append(x)
                image_outputs.append(prediction * m)
            # Write images
            self.write_images(
                image_outputs=image_outputs,
                mode=mode,
                domain=domain,
                task="painter",
                im_per_row=self.opts.comet.im_per_row.get("p", 4),
                rows_per_log=self.opts.comet.get("rows_per_log", 5),
                comet_exp=self.exp,
            )

        return 0

    def log_comet_combined_images(self, mode, domain):

        image_outputs = []
        for im_set in self.display_images[mode][domain]:
            x = im_set["data"]["x"].unsqueeze(0).to(self.device)
            # m = im_set["data"]["m"].unsqueeze(0).to(self.device)

            z = self.sample_z(x.shape[0]) if not self.no_z else None
            m = self.G.decoders["m"](self.G.encode(x))

            prediction = self.G.painter(z, x * (1.0 - m))
            image_outputs.append(x * (1.0 - m))
            image_outputs.append(prediction)
            image_outputs.append(x)
            image_outputs.append(prediction * m)
        # Write images
        self.write_images(
            image_outputs=image_outputs,
            mode=mode,
            domain=domain,
            task="combined",
            im_per_row=self.opts.comet.im_per_row.get("p", 4),
            rows_per_log=self.opts.comet.get("rows_per_log", 5),
            comet_exp=self.exp,
        )

        return 0

    def write_images(
        self,
        image_outputs,
        mode,
        domain,
        task,
        im_per_row=3,
        rows_per_log=5,
        comet_exp=None,
    ):
        """
        Save output image

        Args:
            image_outputs (list(torch.Tensor)): all the images to log
            mode (str): train or val
            domain (str): current domain
            task (str): current task
            im_per_row (int, optional): umber of images to be displayed per row.
                Typically, for a given task: 3 because [input prediction, target].
                Defaults to 3.
            rows_per_log (int, optional): Number of rows (=samples) per uploaded image.
                Defaults to 5.
            comet_exp (comet_ml.Experiment, optional): experiment to use.
                Defaults to None.
        """
        curr_iter = self.logger.global_step
        nb_per_log = im_per_row * rows_per_log
        for logidx in range(rows_per_log):
            print(
                "Creating images for {} {} {} {}/{}".format(
                    mode, domain, task, logidx + 1, rows_per_log
                ),
                end="...",
            )
            ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
            if not ims:
                continue
            ims = torch.stack(ims).squeeze()
            image_grid = vutils.make_grid(
                ims, nrow=im_per_row, normalize=True, scale_each=True
            )
            image_grid = image_grid.permute(1, 2, 0).cpu().numpy()

            if comet_exp is not None:
                print("Uploading...", end="")
                comet_exp.log_image(
                    image_grid,
                    name=f"{mode}_{domain}_{task}_{str(curr_iter)}_#{logidx}",
                    step=curr_iter,
                )
                print("Ok", end="\r", flush=True)

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

            if self.logger.epoch % self.opts.train.save_n_epochs == 0:
                self.save()

    def get_g_loss(self, multi_domain_batch, verbose=0):
        m_loss = p_loss = None

        # For now, always compute "representation loss"
        g_loss = 0

        if "m" in self.opts.tasks:
            m_loss = self.get_masker_loss(multi_domain_batch)
            self.logger.losses.gen.masker = m_loss.item()
            g_loss += m_loss

        if "p" in self.opts.tasks:
            p_loss = self.get_painter_loss(multi_domain_batch)
            self.logger.losses.gen.painter = p_loss.item()
            g_loss += p_loss

        if "m" in self.opts.tasks and "p" in self.opts.tasks:
            mp_loss = self.get_combined_loss(multi_domain_batch)
            g_loss += mp_loss

        assert g_loss != 0 and not isinstance(g_loss, int), "No update in get_g_loss!"

        self.logger.losses.gen.total_loss = g_loss.item()

        return g_loss

    def update_g(self, multi_domain_batch, verbose=0):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

        * compute loss
            * if using Sam Lavoie's representational_training:
                * compute either representation_loss or translation_loss
                  depending on the current step vs opts.train.representation_steps
            * otherwise compute both
        * loss.backward()
        * g_opt_step()
            * g_opt.step() or .extrapolation() depending on self.logger.global_step
        * logs losses on comet.ml with self.log_losses(model_to_update="G")

        Args:
            multi_domain_batch (dict): dictionnary of domain batches
        """
        zero_grad(self.G)
        g_loss = self.get_g_loss(multi_domain_batch, verbose)
        g_loss.backward()
        self.g_opt_step()
        self.log_losses(model_to_update="G", mode="train")

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
        step_loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain == "rf":
                continue

            x = batch["data"]["x"]
            self.z = self.G.encode(x)

            if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                depth_prediction, z_depth = self.G.decoders["d"](self.z)
                z_feat_fusion = self.z * z_depth
            else:
                depth_prediction = None
            # ---------------------------------
            # -----  classifier loss (1)  -----
            # ---------------------------------
            if self.opts.train.latent_domain_adaptation:
                output_classifier = self.C(self.z)

                # Cross entropy loss (with sigmoid) with fake labels to fool C
                update_loss = (
                    self.losses["G"]["classifier"](
                        output_classifier,
                        fake_domains_to_class_tensor(batch["domain"], one_hot),
                    )
                    * lambdas.G.classifier
                )

                step_loss += update_loss
                self.logger.losses.gen.classifier[batch_domain] = update_loss.item()

            # -------------------------------------------------
            # -----  task-specific regression losses (2)  -----
            # -------------------------------------------------
            for update_task, update_target in batch["data"].items():
                if update_task == "d":
                    if self.opts.gen.d.use_dada:
                        scaler = lambdas.G.d.dada
                    else:
                        scaler = lambdas.G.d.main

                    update_loss = (
                        self.losses["G"]["tasks"]["d"](depth_prediction, update_target)
                        * scaler
                    )
                    step_loss += update_loss

                    self.logger.losses.gen.task[update_task][
                        batch_domain
                    ] = update_loss.item()
                elif update_task == "s":
                    if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                        prediction = self.G.decoders[update_task](z_feat_fusion)
                    else:
                        prediction = self.G.decoders[update_task](self.z)

                    # Supervised segmentation loss: crossent for sim domain,
                    # crossent_pseudo for real ; loss is crossent in any case
                    if batch_domain == "s" or self.opts.gen.s.use_pseudo_labels:
                        if batch_domain == "s":
                            loss_name = "crossent"
                        else:
                            loss_name = "crossent_pseudo"

                        update_loss = (
                            self.losses["G"]["tasks"]["s"]["crossent"](
                                prediction, update_target.squeeze(1)
                            )
                            * lambdas.G["s"][loss_name]
                        )
                        step_loss += update_loss

                        self.logger.losses.gen.task["s"][loss_name][
                            batch_domain
                        ] = update_loss.item()
                    if batch_domain == "r":
                        # Entropy minimization loss
                        if self.opts.gen.s.use_minent:
                            # Direct entropy minimization
                            update_loss = (
                                self.losses["G"]["tasks"][update_task]["minent"](
                                    prediction
                                )
                                * lambdas.G[update_task]["minent"]
                            )
                            step_loss += update_loss

                            self.logger.losses.gen.task[update_task]["minent"][
                                batch_domain
                            ] = update_loss.item()

                        # Fool ADVENT discriminator
                        if self.opts.gen.s.use_advent:
                            update_loss = (
                                self.losses["G"]["tasks"][update_task]["advent"](
                                    prediction,
                                    self.domain_labels["s"],
                                    self.D["s"]["Advent"],
                                    depth_prediction,
                                )
                                * lambdas.G[update_task]["advent"]
                            )
                            step_loss += update_loss
                            self.logger.losses.gen.task[update_task]["advent"][
                                batch_domain
                            ] = update_loss.item()
                elif update_task == "m":
                    if "d" in self.opts.tasks and self.opts.gen.m.do_feat_fusion:
                        prediction = self.G.decoders[update_task](z_feat_fusion)
                    else:
                        prediction = self.G.decoders[update_task](self.z)

                    if batch_domain == "s":

                        # Main loss first:
                        update_loss = (
                            self.losses["G"]["tasks"]["m"]["bce"](
                                prediction, update_target
                            )
                            * lambdas.G.m.bce
                        )
                        step_loss += update_loss

                        self.logger.losses.gen.task["m"]["bce"][
                            "s"
                        ] = update_loss.item()

                    # Then TV loss
                    update_loss = (
                        self.losses["G"]["tasks"]["m"]["tv"](prediction)
                        * self.opts.train.lambdas.G.m.tv
                    )
                    step_loss += update_loss

                    self.logger.losses.gen.task["m"]["tv"][
                        batch_domain
                    ] = update_loss.item()

                    # Then GroundIntersection loss
                    if batch_domain == "r":
                        if self.opts.gen.m.use_ground_intersection:
                            if self.verbose > 0:
                                print("Using GroundIntersection loss.")
                            update_loss = (
                                self.losses["G"]["tasks"][update_task]["gi"](
                                    prediction, update_target
                                )
                                * lambdas.G[update_task]["gi"]
                            )
                            step_loss += update_loss
                            self.logger.losses.gen.task[update_task]["gi"][
                                batch_domain
                            ] = update_loss.item()

                    if batch_domain == "r":
                        pred_complementary = 1 - prediction
                        prob = torch.cat([prediction, pred_complementary], dim=1)
                        if self.opts.gen.m.use_minent:
                            # Then Minent loss
                            update_loss = (
                                self.losses["G"]["tasks"]["m"]["minent"](
                                    prob.to(self.device)
                                )
                                * self.opts.train.lambdas.advent.ent_main
                            )
                            step_loss += update_loss
                            self.logger.losses.gen.task["m"]["minent"][
                                "r"
                            ] = update_loss.item()

                        if self.opts.gen.m.use_advent:
                            # Then Advent loss
                            update_loss = (
                                self.losses["G"]["tasks"]["m"]["advent"](
                                    prob.to(self.device),
                                    self.domain_labels["s"],
                                    self.D["m"]["Advent"],
                                )
                                * self.opts.train.lambdas.advent.adv_main
                            )
                            step_loss += update_loss
                            self.logger.losses.gen.task["m"]["advent"][
                                batch_domain
                            ] = update_loss.item()
                elif update_task not in {"m", "p", "x", "s", "d"}:
                    prediction = self.G.decoders[update_task](self.z)

                    update_loss = (
                        self.losses["G"]["tasks"][update_task](
                            prediction, update_target
                        )
                        * lambdas.G[update_task]
                    )

                    step_loss += update_loss
                    self.logger.losses.gen.task[update_task][
                        batch_domain
                    ] = update_loss.item()

        return step_loss

    def sample_z(self, batch_size):
        return torch.empty(
            batch_size,
            self.opts.gen.p.latent_dim,
            self.G.painter.z_h,
            self.G.painter.z_w,
            device=self.device,
        ).normal_(mean=0, std=1.0)

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
        z = self.sample_z(x.shape[0]) if not self.no_z else None
        masked_x = x * (1.0 - m)

        fake_flooded = self.G.painter(z, masked_x)

        update_loss = (
            self.losses["G"]["p"]["vgg"](
                vgg_preprocess(fake_flooded * m), vgg_preprocess(x * m)
            )
            * lambdas.G.p.vgg
        )

        self.logger.losses.gen.p.vgg = update_loss.item()
        step_loss += update_loss

        update_loss = self.losses["G"]["p"]["tv"](fake_flooded * m) * lambdas.G.p.tv
        self.logger.losses.gen.p.tv = update_loss.item()
        step_loss += update_loss

        update_loss = (
            self.losses["G"]["p"]["context"](fake_flooded, x, m) * lambdas.G.p.context
        )
        self.logger.losses.gen.p.context = update_loss.item()
        step_loss += update_loss

        update_loss = (
            self.losses["G"]["p"]["reconstruction"](fake_flooded, x, m)
            * lambdas.G.p.reconstruction
        )
        self.logger.losses.gen.p.reconstruction = update_loss.item()
        step_loss += update_loss

        # GAN Losses
        if self.opts.dis.p.use_local_discriminator:
            fake_d_global = self.D["p"]["global"](fake_flooded)
            fake_d_local = self.D["p"]["local"](fake_flooded * m)

            real_d_global = self.D["p"]["global"](x)

            # Note: discriminator returns [out_1,...,out_num_D] outputs
            # Each out_i is a list [feat1, feat2, ..., pred_i]

            self.logger.losses.gen.p.gan = 0

            update_loss = (
                self.losses["G"]["p"]["gan"](fake_d_global, True, False)
                + self.losses["G"]["p"]["gan"](fake_d_local, True, False)
            ) * lambdas.G["p"]["gan"]

            self.logger.losses.gen.p.gan = update_loss.item()

            step_loss += update_loss

            # Feature matching loss (only on global discriminator)
            # Order must be real, fake
            if self.opts.dis.p.get_intermediate_features:
                update_loss = (
                    self.losses["G"]["p"]["featmatch"](real_d_global, fake_d_global)
                    * lambdas.G["p"]["featmatch"]
                )

                if isinstance(update_loss, float):
                    self.logger.losses.gen.p.featmatch = update_loss
                else:
                    self.logger.losses.gen.p.featmatch = update_loss.item()

                step_loss += update_loss

        else:
            real_fake_d = self.D["p"](
                torch.cat(
                    [torch.cat([m, x], axis=1), torch.cat([m, fake_flooded], axis=1)],
                    axis=0,
                )
            )
            fake_d, real_d = divide_pred(real_fake_d)
            update_loss = self.losses["G"]["p"]["gan"](fake_d, True, False)
            self.logger.losses.gen.p.gan = update_loss.item()
            step_loss += update_loss

            update_loss = (
                self.losses["G"]["p"]["featmatch"](real_d, fake_d)
                * lambdas.G["p"]["featmatch"]
            )

            if isinstance(update_loss, float):
                self.logger.losses.gen.p.featmatch = update_loss
            else:
                self.logger.losses.gen.p.featmatch = update_loss.item()

            step_loss += update_loss

        return step_loss

    def get_combined_loss(self, multi_domain_batch):  # TODO update docstrings
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
        step_loss = 0
        lambdas = self.opts.train.lambdas
        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain == "rf" or batch_domain == "s":
                continue

            x = batch["data"]["x"]
            self.z = self.G.encode(x)

            update_task = "m"
            # Get mask from masker
            m = self.G.decoders[update_task](self.z)

            z = self.sample_z(x.shape[0]) if not self.no_z else None
            masked_x = x * (1.0 - m)

            fake_flooded = self.G.painter(z, masked_x)
            # GAN Losses
            fake_d_global = self.D["p"]["global"](fake_flooded)

            # Note: discriminator returns [out_1,...,out_num_D] outputs
            # Each out_i is a list [feat1, feat2, ..., pred_i]

            self.logger.losses.gen.p.endtoend = 0

            num_D = len(fake_d_global)
            for i in range(num_D):
                # Take last element for GAN loss on discrim prediction
                update_loss = (
                    (self.losses["G"]["p"]["gan"](fake_d_global[i][-1], True))
                    * lambdas.G["p"]["gan"]
                    / num_D
                )

                self.logger.losses.gen.p.endtoend += update_loss.item()

        return step_loss

    def update_d(self, multi_domain_batch, verbose=0):
        # ? split representational as in update_g
        # ? repr: domain-adaptation traduction
        zero_grad(self.D)
        d_loss = self.get_d_loss(multi_domain_batch, verbose)

        d_loss.backward()
        self.d_opt_step()

        self.logger.losses.disc.total_loss = d_loss.item()
        self.log_losses(model_to_update="D", mode="train")

    def get_d_loss(self, multi_domain_batch, verbose=0):
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

        for batch_domain, batch in multi_domain_batch.items():
            x = batch["data"]["x"]
            m = batch["data"]["m"]

            if batch_domain == "rf":
                # sample vector
                with torch.no_grad():
                    # see spade compute_discriminator_loss
                    z_paint = self.sample_z(x.shape[0]) if not self.no_z else None
                    fake = self.G.painter(z_paint, x * (1.0 - m))
                    fake = fake.detach()
                    fake.requires_grad_()

                if self.opts.dis.p.use_local_discriminator:
                    fake_d_global = self.D["p"]["global"](fake)
                    real_d_global = self.D["p"]["global"](x)
                    fake_d_local = self.D["p"]["local"](fake * m)
                    real_d_local = self.D["p"]["local"](x * m)

                    global_loss = self.losses["D"]["p"](
                        fake_d_global, False, True
                    ) + self.losses["D"]["p"](real_d_global, True, True)

                    local_loss = self.losses["D"]["p"](
                        fake_d_local, False, True
                    ) + self.losses["D"]["p"](real_d_local, True, True)

                    disc_loss["p"]["global"] += global_loss
                    disc_loss["p"]["local"] += local_loss
                else:
                    real_fake_d = self.D["p"](
                        torch.cat(
                            [torch.cat([m, x], axis=1), torch.cat([m, fake], axis=1)],
                            axis=0,
                        )
                    )
                    fake_d, real_d = divide_pred(real_fake_d)
                    disc_loss["p"]["gan"] = self.losses["D"]["p"](fake_d, False, True)
                    disc_loss["p"]["gan"] += self.losses["D"]["p"](real_d, True, True)

                # Note: discriminator returns [out_1,...,out_num_D] outputs
                # Each out_i is a list [feat1, feat2, ..., pred_i]

            else:
                z = self.G.encode(x)

                if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                    depth_prediction, z_depth = self.G.decoders["d"](z)
                    depth_prediction.detach()
                    z_feat_fusion = z * z_depth

                if "m" in self.opts.tasks:
                    if self.opts.gen.m.use_advent:
                        if verbose > 0:
                            print("Now training the ADVENT discriminator!")
                        if "d" in self.opts.tasks and self.opts.gen.m.do_feat_fusion:
                            fake_mask = self.G.decoders["m"](z_feat_fusion)
                        else:
                            fake_mask = self.G.decoders["m"](z)
                        fake_complementary_mask = 1 - fake_mask
                        prob = torch.cat([fake_mask, fake_complementary_mask], dim=1)
                        prob = prob.detach()

                        loss_main = self.losses["D"]["advent"](
                            prob.to(self.device),
                            self.domain_labels[batch_domain],
                            self.D["m"]["Advent"],
                        )
                        if self.opts.dis.m.gan_type == "GAN" or "WGAN_norm":
                            disc_loss["m"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                            )
                        elif self.opts.dis.m.gan_type == "WGAN":
                            for p in self.D["m"]["Advent"].parameters():
                                p.data.clamp_(
                                    self.opts.dis.m.wgan_clamp_lower,
                                    self.opts.dis.m.wgan_clamp_upper,
                                )
                            disc_loss["m"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                            )
                        elif self.opts.dis.m.gan_type == "WGAN_gp":
                            prob_need_grad = autograd.Variable(prob, requires_grad=True)
                            d_out = self.D["m"]["Advent"](prob_need_grad)
                            gp = get_WGAN_gradient(prob_need_grad, d_out)
                            disc_loss["m"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                                + self.opts.train.lambdas.advent.WGAN_gp * gp
                            )
                        else:
                            raise NotImplementedError
                if "s" in self.opts.tasks:
                    if self.opts.gen.s.use_advent:
                        if "d" in self.opts.tasks and self.opts.gen.d.use_dada:
                            depth_preds = depth_prediction
                            preds = self.G.decoders["s"](z_feat_fusion)
                        else:
                            depth_preds = None
                            preds = self.G.decoders["s"](z)
                        preds = preds.detach()

                        loss_main = self.losses["D"]["advent"](
                            preds.to(self.device),
                            self.domain_labels[batch_domain],
                            self.D["s"]["Advent"],
                            depth_preds,
                        )

                        if self.opts.dis.s.gan_type == "GAN" or "WGAN_norm":
                            disc_loss["s"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                            )
                        elif self.opts.dis.s.gan_type == "WGAN":
                            for p in self.D["s"]["Advent"].parameters():
                                p.data.clamp_(
                                    self.opts.dis.s.wgan_clamp_lower,
                                    self.opts.dis.s.wgan_clamp_upper,
                                )
                            disc_loss["s"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                            )
                        elif self.opts.dis.s.gan_type == "WGAN_gp":
                            prob_need_grad = autograd.Variable(prob, requires_grad=True)
                            d_out = self.D["s"]["Advent"](prob_need_grad)
                            gp = get_WGAN_gradient(prob_need_grad, d_out)
                            disc_loss["s"]["Advent"] += (
                                self.opts.train.lambdas.advent.adv_main * loss_main
                                + self.opts.train.lambdas.advent.WGAN_gp * gp
                            )
                        else:
                            raise NotImplementedError

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

    def update_c(self, multi_domain_batch):
        """
        Update the classifier using normal labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
                the trainer's loaders

        """
        zero_grad(self.C)
        c_loss = self.get_classifier_loss(multi_domain_batch)
        # ? Log policy
        self.logger.losses.classifier = c_loss.item()
        c_loss.backward()
        self.c_opt_step()

    def get_classifier_loss(self, multi_domain_batch):
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
            self.z = self.G.encode(batch["data"]["x"])
            # Forward through classifier, output classifier = (batch_size, 4)
            output_classifier = self.C(self.z)
            # Cross entropy loss (with sigmoid)
            update_loss = self.losses["C"](
                output_classifier,
                domains_to_class_tensor(batch["domain"], one_hot).to(self.device),
            )
            loss += update_loss

        return lambdas.C * loss

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
            self.get_g_loss(multi_domain_batch, verbose)

            if val_logger is None:
                val_logger = deepcopy(self.logger.losses.generator)
            else:
                val_logger = sum_dict(val_logger, self.logger.losses.generator)

        val_logger = div_dict(val_logger, nb_of_batches)
        self.logger.losses.generator = val_logger
        self.log_losses(model_to_update="G", mode="val")

        for d in self.opts.domains:
            self.log_comet_images("train", d)
            self.log_comet_images("val", d)

        if "m" in self.opts.tasks and "p" in self.opts.tasks:
            self.log_comet_combined_images("train", "r")
            self.log_comet_combined_images("val", "r")

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

    def save(self):
        save_dir = Path(self.opts.output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = Path("latest_ckpt.pth")
        save_path = save_dir / save_path

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

        torch.save(save_dict, save_path)

    def resume(self, inference=False):
        # load_path = self.get_latest_ckpt()

        # ----------------------------------------
        # -----  Masker and Painter Loading  -----
        # ----------------------------------------
        tpu = "xla" in str(self.device)
        if tpu:
            print("Resuming on TPU:", self.device)
        if "m" in self.opts.tasks and "p" in self.opts.tasks:
            m_path = self.opts.load_paths.m
            p_path = self.opts.load_paths.p

            if m_path == "none":
                m_path = self.opts.output_path
            if p_path == "none":
                p_path = self.opts.output_path

            # Merge the dicts
            m_ckpt_path = Path(m_path) / Path("checkpoints/latest_ckpt.pth")
            p_ckpt_path = Path(p_path) / Path("checkpoints/latest_ckpt.pth")

            m_checkpoint = torch.load(
                m_ckpt_path, map_location=self.device if not tpu else "cpu"
            )
            p_checkpoint = torch.load(
                p_ckpt_path, map_location=self.device if not tpu else "cpu"
            )
            checkpoint = merge(m_checkpoint, p_checkpoint)
            if tpu:
                checkpoint = xm.send_cpu_data_to_device(checkpoint, self.device)
            print(f"Resuming model from \n  -{m_ckpt_path} \nand \n  -{p_ckpt_path}")
        # ----------------------------------
        # -----  Single Model Loading  -----
        # ----------------------------------
        else:
            load_path = Path(self.opts.output_path) / Path(
                "checkpoints/latest_ckpt.pth"
            )
            checkpoint = torch.load(
                load_path, map_location=self.device if not tpu else "cpu"
            )
            if tpu:
                checkpoint = xm.send_cpu_data_to_device(checkpoint, self.device)
            print(f"Resuming model from {load_path}")

        # -----------------------
        # -----  Restore G  -----
        # -----------------------
        self.G.load_state_dict(checkpoint["G"])

        if inference:
            # only G is needed to infer
            print("Done loading checkpoints.")
            return

        if not ("m" in self.opts.tasks and "p" in self.opts.tasks):
            self.g_opt.load_state_dict(checkpoint["g_opt"])
        self.logger.epoch = checkpoint["epoch"] + 1
        self.logger.global_step = checkpoint["step"]

        # resume scheduler:
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
            if not ("m" in self.opts.tasks and "p" in self.opts.tasks):
                self.d_opt.load_state_dict(checkpoint["d_opt"])

        # -----------------------
        # -----  Restore C  -----
        # -----------------------
        if self.C is not None and get_num_params(self.C) > 0:
            self.C.load_state_dict(checkpoint["C"])
            if not ("m" in self.opts.tasks and "p" in self.opts.tasks):
                self.c_opt.load_state_dict(checkpoint["c_opt"])

        # ---------------------------
        # -----  Resore logger  -----
        # ---------------------------
        self.logger.epoch = checkpoint["epoch"]
        self.logger.global_step = checkpoint["step"]
        # Round step to even number for extraGradient
        if self.logger.global_step % 2 != 0:
            self.logger.global_step += 1

    def get_latest_ckpt(self):
        load_dir = Path(self.opts.output_path) / Path("checkpoints")
        ckpts = os.listdir(str(load_dir.resolve()))
        max_epoch = 0
        max_ckpt = ""
        for ckpt in ckpts:
            ckpt = Path(ckpt)
            epoch = int(ckpt.stem.split("_")[-1])
            if epoch > max_epoch:
                max_epoch = epoch
                max_ckpt = ckpt
        return Path(self.opts.output_path) / Path("checkpoints") / max_ckpt

    def eval_images(self, mode, domain):
        metrics = {"accuracy": accuracy, "iou": iou}
        metric_avg_scores = {}
        for key in metrics.keys():
            metric_avg_scores[key] = 0.0
        if domain != "rf":
            for im_set in self.display_images[mode][domain]:
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)
                m = im_set["data"]["m"].unsqueeze(0).detach().cpu().numpy()
                z = self.G.encode(x)
                pred_mask = self.G.decoders["m"](z).detach().cpu().numpy()
                # Binarize mask
                pred_mask[pred_mask > 0.5] = 1.0

                for metric_key in metrics.keys():
                    metric_score = metrics[metric_key](pred_mask, m)
                    metric_avg_scores[metric_key] += metric_score / len(
                        self.display_images[mode][domain]
                    )

            if self.exp is not None:
                self.exp.log_metrics(
                    metric_avg_scores,
                    prefix=f"metrics_{mode}",
                    step=self.logger.global_step,
                )

        return 0
