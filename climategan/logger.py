from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from addict import Dict
from PIL import Image
from torch.nn.functional import interpolate, sigmoid

from climategan.data import decode_segmap_merged_labels
from climategan.tutils import (
    all_texts_to_tensors,
    decode_bucketed_depth,
    normalize_tensor,
    write_architecture,
)
from climategan.utils import flatten_opts


class Logger:
    def __init__(self, trainer):
        self.losses = Dict()
        self.time = Dict()
        self.trainer = trainer
        self.global_step = 0
        self.epoch = 0

    def log_comet_images(self, mode, domain, minimal=False, all_only=False):
        trainer = self.trainer
        save_images = {}
        all_images = []
        n_all_ims = None
        all_legends = ["Input"]
        task_legends = {}

        if domain not in trainer.display_images[mode]:
            return

        # --------------------
        # -----  Masker  -----
        # --------------------
        n_ims = len(trainer.display_images[mode][domain])
        print(" " * 60, end="\r")
        if domain != "rf":
            for j, display_dict in enumerate(trainer.display_images[mode][domain]):

                print(f"Inferring sample {mode} {domain} {j+1}/{n_ims}", end="\r")

                x = display_dict["data"]["x"].unsqueeze(0).to(trainer.device)
                z = trainer.G.encode(x)

                s_pred = decoded_s_pred = d_pred = z_depth = None
                for k, task in enumerate(["d", "s", "m"]):

                    if (
                        task not in display_dict["data"]
                        or task not in trainer.opts.tasks
                    ):
                        continue

                    task_legend = ["Input"]
                    target = display_dict["data"][task]
                    target = target.unsqueeze(0).to(trainer.device)
                    task_saves = []

                    if task not in save_images:
                        save_images[task] = []

                    prediction = None
                    if task == "m":
                        cond = None
                        if s_pred is not None and d_pred is not None:
                            cond = trainer.G.make_m_cond(d_pred, s_pred, x)

                        prediction = trainer.G.decoders[task](z, cond, z_depth)
                    elif task == "d":
                        prediction, z_depth = trainer.G.decoders[task](z)
                    elif task == "s":
                        prediction = trainer.G.decoders[task](z, z_depth)

                    if task == "s":
                        # Log fire
                        wildfire_tens = trainer.compute_fire(x, prediction)
                        task_saves.append(wildfire_tens)
                        task_legend.append("Wildfire")
                        # Log seg output
                        s_pred = prediction.clone()
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
                        decoded_s_pred = prediction
                        task_saves.append(target)
                        task_legend.append("Target Segmentation")

                    elif task == "m":
                        prediction = sigmoid(prediction).repeat(1, 3, 1, 1)
                        task_saves.append(x * (1.0 - prediction))
                        if not minimal:
                            task_saves.append(
                                x * (1.0 - (prediction > 0.1).to(torch.int))
                            )
                            task_saves.append(
                                x * (1.0 - (prediction > 0.5).to(torch.int))
                            )

                        task_saves.append(x * (1.0 - target.repeat(1, 3, 1, 1)))
                        task_legend.append("Masked input")

                        if not minimal:
                            task_legend.append("Masked input (>0.1)")
                            task_legend.append("Masked input (>0.5)")

                        task_legend.append("Masked input (target)")
                        # dummy pixels to fool scaling and preserve mask range
                        prediction[:, :, 0, 0] = 1.0
                        prediction[:, :, -1, -1] = 0.0

                    elif task == "d":
                        # prediction is a log depth tensor
                        d_pred = prediction
                        target = normalize_tensor(target) * 255
                        if prediction.shape[1] > 1:
                            prediction = decode_bucketed_depth(
                                prediction, self.trainer.opts
                            )
                        smogged = self.trainer.compute_smog(
                            x, d=prediction, s=decoded_s_pred, use_sky_seg=False
                        )
                        prediction = normalize_tensor(prediction)
                        prediction = prediction.repeat(1, 3, 1, 1)
                        task_saves.append(smogged)
                        task_legend.append("Smogged")
                        task_saves.append(target.repeat(1, 3, 1, 1))
                        task_legend.append("Depth target")

                    task_saves.append(prediction)
                    task_legend.append(f"Predicted {task}")

                    save_images[task].append(x.cpu().detach())
                    if k == 0:
                        all_images.append(save_images[task][-1])

                    task_legends[task] = task_legend
                    if j == 0:
                        all_legends += task_legend[1:]

                    for im in task_saves:
                        save_images[task].append(im.cpu().detach())
                        all_images.append(save_images[task][-1])

                if j == 0:
                    n_all_ims = len(all_images)

            if not all_only:
                for task in save_images.keys():
                    # Write images:
                    self.upload_images(
                        image_outputs=save_images[task],
                        mode=mode,
                        domain=domain,
                        task=task,
                        im_per_row=trainer.opts.comet.im_per_row.get(task, 4),
                        rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
                        legends=task_legends[task],
                    )

            if len(save_images) > 1:
                self.upload_images(
                    image_outputs=all_images,
                    mode=mode,
                    domain=domain,
                    task="all",
                    im_per_row=n_all_ims,
                    rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
                    legends=all_legends,
                )
        # ---------------------
        # -----  Painter  -----
        # ---------------------
        else:
            # in the rf domain display_size may be different from fid.n_images
            limit = trainer.opts.comet.display_size
            image_outputs = []
            legends = []
            for im_set in trainer.display_images[mode][domain][:limit]:
                x = im_set["data"]["x"].unsqueeze(0).to(trainer.device)
                m = im_set["data"]["m"].unsqueeze(0).to(trainer.device)

                prediction = trainer.G.paint(m, x)

                image_outputs.append(x * (1.0 - m))
                image_outputs.append(prediction)
                image_outputs.append(x)
                image_outputs.append(prediction * m)
                if not legends:
                    legends.append("Masked Input")
                    legends.append("Painted Input")
                    legends.append("Input")
                    legends.append("Isolated Water")
            # Write images
            self.upload_images(
                image_outputs=image_outputs,
                mode=mode,
                domain=domain,
                task="painter",
                im_per_row=trainer.opts.comet.im_per_row.get("p", 4),
                rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
                legends=legends,
            )

        return 0

    def log_losses(self, model_to_update="G", mode="train"):
        """Logs metrics on comet.ml

        Args:
            model_to_update (str, optional): One of "G", "D". Defaults to "G".
        """
        trainer = self.trainer
        loss_names = {"G": "gen", "D": "disc"}

        if trainer.opts.train.log_level < 1:
            return

        if trainer.exp is None:
            return

        assert model_to_update in {
            "G",
            "D",
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
        legends = []
        im_per_row = 0
        for i, im_set in enumerate(trainer.display_images[mode][domain]):
            x = im_set["data"]["x"].unsqueeze(0).to(trainer.device)
            # m = im_set["data"]["m"].unsqueeze(0).to(trainer.device)

            m = trainer.G.mask(x=x)
            m_bin = (m > 0.5).to(m.dtype)
            prediction = trainer.G.paint(m, x)
            prediction_bin = trainer.G.paint(m_bin, x)

            image_outputs.append(x)
            legends.append("Input")
            image_outputs.append(x * (1.0 - m))
            legends.append("Soft Masked Input")
            image_outputs.append(prediction)
            legends.append("Painted")
            image_outputs.append(prediction * m)
            legends.append("Soft Masked Painted")
            image_outputs.append(x * (1.0 - m_bin))
            legends.append("Binary (0.5) Masked Input")
            image_outputs.append(prediction_bin)
            legends.append("Binary (0.5) Painted")
            image_outputs.append(prediction_bin * m_bin)
            legends.append("Binary (0.5) Masked Painted")

            if i == 0:
                im_per_row = len(image_outputs)
        # Upload images
        self.upload_images(
            image_outputs=image_outputs,
            mode=mode,
            domain=domain,
            task="combined",
            im_per_row=im_per_row or 7,
            rows_per_log=trainer.opts.comet.get("rows_per_log", 5),
            legends=legends,
        )

        return 0

    def upload_images(
        self,
        image_outputs,
        mode,
        domain,
        task,
        im_per_row=3,
        rows_per_log=5,
        legends=[],
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
        n_logs = len(image_outputs) // nb_per_log + 1

        header = None
        if len(legends) == im_per_row and all(isinstance(t, str) for t in legends):
            header_width = max(im.shape[-1] for im in image_outputs)
            headers = all_texts_to_tensors(legends, width=header_width)
            header = torch.cat(headers, dim=-1)

        for logidx in range(n_logs):
            print(" " * 100, end="\r", flush=True)
            print(
                "Uploading images for {} {} {} {}/{}".format(
                    mode, domain, task, logidx + 1, n_logs
                ),
                end="...",
                flush=True,
            )
            ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
            if not ims:
                continue

            ims = self.upsample(ims)
            ims = torch.stack([im.squeeze() for im in ims]).squeeze()
            image_grid = vutils.make_grid(
                ims, nrow=im_per_row, normalize=True, scale_each=True, padding=0
            )

            if header is not None:
                image_grid = torch.cat(
                    [header.to(image_grid.device), image_grid], dim=1
                )

            image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
            trainer.exp.log_image(
                Image.fromarray((image_grid * 255).astype(np.uint8)),
                name=f"{mode}_{domain}_{task}_{str(curr_iter)}_#{logidx}",
                step=curr_iter,
            )

    def upsample(self, ims):
        h = max(im.shape[-2] for im in ims)
        w = max(im.shape[-1] for im in ims)
        new_ims = []
        for im in ims:
            im = interpolate(im, (h, w), mode="bilinear")
            new_ims.append(im)
        return new_ims

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

    def log_architecture(self):
        write_architecture(self.trainer)

        if self.trainer.exp is None:
            return

        for f in Path(self.trainer.opts.output_path).glob("archi*.txt"):
            self.trainer.exp.log_asset(str(f), overwrite=True)
