from addict import Dict
import torchvision.utils as vutils

import numpy as np
import torch
from torch.nn.functional import sigmoid
from omnigan.data import decode_segmap_merged_labels
from omnigan.tutils import normalize_tensor
from omnigan.utils import flatten_opts
from PIL import Image


class Logger:
    def __init__(self, trainer):
        self.losses = Dict()
        self.time = Dict()
        self.trainer = trainer
        self.global_step = 0
        self.epoch = 0

    def log_comet_images(self, mode, domain):
        trainer = self.trainer
        save_images = {}
        # --------------------
        # -----  Masker  -----
        # --------------------
        if domain != "rf":
            for j, display_dict in enumerate(trainer.display_images[mode][domain]):
                print(j, end="\r")
                x = display_dict["data"]["x"].unsqueeze(0).to(trainer.device)
                z = trainer.G.encode(x)

                seg_pred = None
                for task in sorted(self.trainer.opts.tasks, reverse=True):
                    target = display_dict["data"][task]
                    target = target.unsqueeze(0).to(trainer.device)
                    task_saves = []

                    if task == "p":
                        continue

                    if task not in save_images:
                        save_images[task] = []

                    prediction = trainer.G.decoders[task](z)

                    if task == "s":
                        target = (
                            decode_segmap_merged_labels(target, domain, True)
                            .float()
                            .to(trainer.device)
                        )
                        prediction = (
                            decode_segmap_merged_labels(prediction, domain, False)
                            .float()
                            .to(trainer.device)
                        )
                        seg_pred = prediction
                        task_saves.append(target)

                    elif task == "m":
                        prediction = sigmoid(prediction).repeat(1, 3, 1, 1)
                        task_saves.append(x * (1.0 - prediction))
                        task_saves.append(x * (1.0 - (prediction > 0.1).to(torch.int)))
                        task_saves.append(x * (1.0 - (prediction > 0.5).to(torch.int)))
                        task_saves.append(x * (1.0 - target.repeat(1, 3, 1, 1)))
                        # dummy pixels to fool scaling and preserve mask range
                        prediction[:, :, 0, 0] = 1.0
                        prediction[:, :, -1, -1] = 0.0

                    elif task == "d":
                        # prediction is a log depth tensor
                        target = normalize_tensor(target) * 255
                        prediction = normalize_tensor(prediction) * 255
                        smogged = self.trainer.compute_smog(
                            x, d=prediction, s=seg_pred, use_sky_seg=False
                        )
                        prediction = prediction.repeat(1, 3, 1, 1)
                        task_saves.append(target.repeat(1, 3, 1, 1))
                        task_saves.append(smogged)

                    task_saves.append(prediction)
                    save_images[task].append(x.cpu().detach())

                    for im in task_saves:
                        save_images[task].append(im.cpu().detach())

            for task in save_images.keys():
                # Write images:
                self.write_images(
                    image_outputs=save_images[task],
                    mode=mode,
                    domain=domain,
                    task=task,
                    im_per_row=trainer.opts.comet.im_per_row.get(task, 4),
                    rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
                )
        # ---------------------
        # -----  Painter  -----
        # ---------------------
        else:
            # in the rf domain display_size may be different from fid.n_images
            limit = trainer.opts.comet.display_size
            image_outputs = []
            for im_set in trainer.display_images[mode][domain][:limit]:
                x = im_set["data"]["x"].unsqueeze(0).to(trainer.device)
                m = im_set["data"]["m"].unsqueeze(0).to(trainer.device)

                prediction = trainer.G.paint(m, x)

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
                im_per_row=trainer.opts.comet.im_per_row.get("p", 4),
                rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
            )

        return 0

    def log_losses(self, model_to_update="G", mode="train"):
        """Logs metrics on comet.ml

        Args:
            model_to_update (str, optional): One of "G", "D" or "C". Defaults to "G".
        """
        trainer = self.trainer
        loss_names = {"G": "gen", "D": "disc", "C": "classifier"}

        if trainer.opts.train.log_level < 1:
            return

        if trainer.exp is None:
            return

        assert model_to_update in {
            "G",
            "D",
            "C",
        }, "unknown model to log losses {}".format(model_to_update)

        loss_to_update = self.losses[loss_names[model_to_update]]

        losses = loss_to_update.copy()

        if trainer.opts.train.log_level == 1:
            # Only log aggregated losses: delete other keys in losses
            for k in loss_to_update:
                if k not in {"masker", "total_loss", "painter"}:
                    del losses[k]
        # convert losses into a single-level dictionnary

        losses = flatten_opts(losses)
        trainer.exp.log_metrics(
            losses, prefix=f"{model_to_update}_{mode}", step=self.global_step
        )

    def log_learning_rates(self):
        if self.trainer.exp is None:
            return
        lrs = {}
        trainer = self.trainer
        if trainer.g_scheduler is not None:
            for name, lr in zip(
                trainer.lr_names["G"], trainer.g_scheduler.get_last_lr()
            ):
                lrs[f"lr_G_{name}"] = lr
        if trainer.d_scheduler is not None:
            for name, lr in zip(
                trainer.lr_names["D"], trainer.d_scheduler.get_last_lr()
            ):
                lrs[f"lr_D_{name}"] = lr
        if trainer.c_scheduler is not None:
            for name, lr in zip(
                trainer.lr_names["C"], trainer.c_scheduler.get_last_lr()
            ):
                lrs[f"lr_C_{name}"] = lr
        trainer.exp.log_metrics(lrs, step=self.global_step)

    def log_step_time(self, time):
        """Logs step-time on comet.ml

        Args:
            step_time (float): step-time in seconds
        """
        if self.trainer.exp:
            self.trainer.exp.log_metric(
                "step-time", time - self.time.step_start, step=self.global_step
            )

    def log_epoch_time(self, time):
        """Logs step-time on comet.ml

        Args:
            step_time (float): step-time in seconds
        """
        if self.trainer.exp:
            self.trainer.exp.log_metric(
                "epoch-time", time - self.time.epoch_start, step=self.global_step
            )

    def log_comet_combined_images(self, mode, domain):

        trainer = self.trainer
        image_outputs = []
        for im_set in trainer.display_images[mode][domain]:
            x = im_set["data"]["x"].unsqueeze(0).to(trainer.device)
            # m = im_set["data"]["m"].unsqueeze(0).to(trainer.device)

            m = trainer.G.mask(x=x)
            prediction = trainer.G.paint(m, x)

            image_outputs.append(x)
            image_outputs.append(x * (1.0 - m))
            image_outputs.append(prediction)
            image_outputs.append(prediction * m)
        # Write images
        self.write_images(
            image_outputs=image_outputs,
            mode=mode,
            domain=domain,
            task="combined",
            im_per_row=trainer.opts.comet.im_per_row.get("p", 4),
            rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
        )

        return 0

    def write_images(
        self, image_outputs, mode, domain, task, im_per_row=3, rows_per_log=5,
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
        trainer = self.trainer
        if trainer.exp is None:
            return
        curr_iter = self.global_step
        nb_per_log = im_per_row * rows_per_log
        for logidx in range(rows_per_log):
            print(" " * 100, end="\r", flush=True)
            print(
                "Creating images for {} {} {} {}/{}".format(
                    mode, domain, task, logidx + 1, rows_per_log
                ),
                end="...",
                flush=True,
            )
            ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
            if not ims:
                continue

            ims = self.padd(ims)
            ims = torch.stack([im.squeeze() for im in ims]).squeeze()
            image_grid = vutils.make_grid(
                ims, nrow=im_per_row, normalize=True, scale_each=True
            )
            image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
            trainer.exp.log_image(
                Image.fromarray((image_grid * 255).astype(np.uint8)),
                name=f"{mode}_{domain}_{task}_{str(curr_iter)}_#{logidx}",
                step=curr_iter,
            )

    def padd(self, ims):
        h = max(im.shape[-2] for im in ims)
        w = max(im.shape[-1] for im in ims)
        new_ims = []
        for im in ims:
            ih = im.shape[-2]
            iw = im.shape[-1]
            if ih != h or iw != w:
                padded = torch.zeros(im.shape[-3], h, w)
                padded[
                    :, (h - ih) // 2 : (h + ih) // 2, (w - iw) // 2 : (w + iw) // 2
                ] = im
                new_ims.append(padded)
            else:
                new_ims.append(im)

        return new_ims
