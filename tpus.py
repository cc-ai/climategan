import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms as trsfs

from omnigan.data import tensor_loader
from omnigan.trainer import Trainer
from omnigan.utils import load_opts

import torch_xla
import torch_xla.core.xla_model as xm


class Timer:
    def __init__(self, name="", store=None):
        self.name = name
        self.store = store

    def __enter__(self):
        """Start a new timer as a context manager"""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        t = time.perf_counter()
        new_time = t - self._start_time

        if self.store is not None:
            assert isinstance(self.store, list)
            self.store.append(new_time)
        if self.name:
            print(f"[{self.name}] Elapsed time: {new_time:.3f}")


def gpu_used():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return "{:.2f}GB".format(info.used / 1e9)


def isimg(path_file):
    if (
        path_file.suffix == ".jpg"
        or path_file.suffix == ".png"
        or path_file.suffix == ".PNG"
        or path_file.suffix == ".JPG"
    ):
        return True
    else:
        return False


def prepare_image(img_tensor, new_size, transforms, device, use_half):
    img_tensor = F.interpolate(img_tensor, (new_size, new_size), mode="nearest")
    img_tensor = img_tensor.squeeze(0)
    for tf in transforms:
        img_tensor = tf(img_tensor)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    if use_half:
        img_tensor = img_tensor.half()
    return img_tensor


def prepare_mask(mask_tensor, device, use_half):
    mask_tensor = mask_tensor.squeeze().unsqueeze(0).to(device)
    if use_half:
        mask_tensor = mask_tensor.half()
    return mask_tensor


def eval_folder(
    path_to_images,
    path_to_masks,
    output_dir,
    masker,
    paint,
    opts,
    batch_size,
    use_half,
    trainer,
    device,
    save_images,
    empty_cuda_cache,
    loaded_images,
):
    if empty_cuda_cache:
        torch.cuda.empty_cache()
    model = trainer.G
    if use_half:
        model = model.half()
    model.eval()

    if not loaded_images:
        image_list = os.listdir(path_to_images)
        image_list.sort()
        images = [
            tensor_loader(path_to_images / Path(i), task="x", domain="val")
            for i in image_list
        ]
    else:
        images = loaded_images

    if not masker:
        mask_list = os.listdir(path_to_masks)
        mask_list.sort()
        masks = [
            tensor_loader(
                path_to_masks / Path(i), task="m", domain="val", binarize=False
            )
            for i in mask_list
        ]

    painter_inference_time = []
    masker_inference_time = []
    full_procedure_time = []
    inference_loop_time = []

    output_dir = output_dir / f"bs_{batch_size}{'_half' if use_half else ''}"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Batch Size:", batch_size)
    print("Using Half:", use_half)

    with Timer(
        "Full procedure on {} images".format(len(images)), store=full_procedure_time
    ):

        if not masker:
            mask_tensors = [prepare_mask(m, device, use_half) for m in masks]

        image_tensors = [
            prepare_image(im, new_size, transforms, device, use_half) for im in images
        ]

        with Timer("Inference loop", store=inference_loop_time):
            for i in range(len(image_tensors) // batch_size + 1):
                img = image_tensors[i * batch_size : (i + 1) * batch_size]
                if not img:
                    continue
                img = torch.cat(img, axis=0)
                print("Batch", i, img.shape, img.device, end="\r", flush=True)

                if not masker:
                    mask = mask_tensors[i * batch_size : (i + 1) * batch_size]
                    img = torch.cat(img, axis=0)

                if masker:
                    if "m2" in opts.tasks:
                        z = model.encode(img)
                        num_masks = 10
                        label_vals = np.linspace(start=0, stop=1, num=num_masks)
                        for label_val in label_vals:
                            z_aug = torch.cat(
                                (
                                    z,
                                    label_val
                                    * trainer.label_2[0, :, :, :].unsqueeze(0),
                                ),
                                dim=1,
                            )
                            mask = model.decoders["m"](z_aug)

                            vutils.save_image(
                                mask,
                                output_dir / (f"mask_{label_val}_" + img_path.name),
                                normalize=True,
                            )
                            for k, (im, m) in enumerate(zip(list(img), list(mask))):
                                if apply_mask and save_images:
                                    vutils.save_image(
                                        im * (1.0 - m) + m,
                                        output_dir
                                        / (
                                            images[i * batch_size + k].stem
                                            + f"img_masked_{label_val}"
                                            + ".jpg"
                                        ),
                                        normalize=True,
                                    )

                    else:
                        with Timer(store=masker_inference_time):
                            z = model.encode(img)
                            mask = model.decoders["m"](z)
                        if save_images:
                            for k, m in enumerate(list(mask)):
                                vutils.save_image(
                                    m,
                                    output_dir
                                    / ("mask_" + images[i * batch_size + k].name),
                                    normalize=True,
                                )

                if paint:
                    with Timer(store=painter_inference_time):
                        z_painter = trainer.sample_z(1)
                        if use_half:
                            z_painter = z_painter.half()

                        fake_flooded = model.painter(z_painter, img * (1.0 - mask))
                        fake_cpu = fake_flooded.cpu().numpy()

                    if save_images:
                        for k, ff in enumerate(list(fake_flooded)):
                            vutils.save_image(
                                ff,
                                output_dir / images[i * batch_size + k].name,
                                normalize=True,
                            )

                    if apply_mask and save_images:
                        for k, (im, m) in enumerate(zip(list(img), list(mask))):
                            vutils.save_image(
                                im * (1.0 - m) + m,
                                output_dir
                                / (
                                    images[i * batch_size + k].stem + "_masked" + ".jpg"
                                ),
                                normalize=True,
                            )
            print()

    print(
        "[Masker]  Average time: {:.3f}s (+/- {:.3f}s)".format(
            np.mean(masker_inference_time), np.std(masker_inference_time)
        )
    )
    print(
        "[Painter] Average time: {:.3f}s (+/- {:.3f}s)".format(
            np.mean(painter_inference_time), np.std(painter_inference_time)
        )
    )
    # print("Used Memory:", gpu_used())

    return (
        painter_inference_time,
        masker_inference_time,
        full_procedure_time[0],
        inference_loop_time[0],
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--masker_dir", required=True, type=str)
    parser.add_argument("-p", "--painter_dir", required=True, type=str)
    parser.add_argument("-d", "--inference_data_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-l", "--deeplab_pretrained", required=True, type=str)
    args = parser.parse_args()
    print(args)

    # -----------------------
    # -----  Load opts  -----
    # -----------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    masker_path = Path(args.masker_dir)

    opts = load_opts(masker_path / "opts.yaml", default="shared/trainer/defaults.yaml")

    opts.tasks = ["m", "p"]
    opts.load_paths.m = str(masker_path)
    opts.load_paths.p = args.painter_dir
    opts.train.resume = True
    opts.output_path = str(masker_path)
    opts.gen.deeplabv2.pretrained_model = args.deeplab_pretrained

    new_size = 640

    paint = False
    masker = False
    if "p" in opts.tasks:
        paint = True
    if "m" in opts.tasks:
        masker = True

    # ------------------------
    # ----- Define model -----
    # ------------------------
    opts.gen.p.latent_dim = 640
    torch.set_grad_enabled(False)
    device = xm.xla_device()
    trainer = Trainer(opts, device=device)
    trainer.input_shape = (3, 640, 640)
    trainer.setup(True)
    trainer.resume(True)

    # -------------------------------
    # -----  Transforms images  -----
    # -------------------------------

    transforms = [trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------

    # eval_folder(args.path_to_images, output_dir)

    # rootdir = Path("/network/tmp1/ccai/data/psychology_wave4")
    rootdir = Path(args.inference_data_dir)

    # eval_folder params
    path_to_images = rootdir  # a folder with a list of images
    path_to_masks = rootdir  # not used if using the masker, otherwise a path to matching masks to the images
    apply_mask = (
        True  # save images with the mask only, in addition to the painted images
    )
    save_images = False  # write the outputs to a folder
    empty_cuda_cache = True  # faster if False but will give erroneous memory footprint
    USE_RTX8000 = True  # Are you with a machine with 48GB?
    loaded_images = None  # will be overloaded with data if preload_images is True
    preload_images = True  # faster if running eval_folder multiple times

    if preload_images:
        image_list = os.listdir(path_to_images)
        image_list.sort()
        loaded_images = [
            tensor_loader(path_to_images / Path(i), task="x", domain="val")
            for i in image_list
        ]

    print("FP32")
    for bs in [1, 2, 4, 8, 16, 32]:
        if bs > 16 and not USE_RTX8000:
            continue
        times = eval_folder(
            path_to_images,
            path_to_masks,
            output_dir,
            masker,
            paint,
            opts,
            bs,
            False,
            trainer,
            device,
            save_images,
            empty_cuda_cache,
            loaded_images,
        )
        print()

    print("FP16 (half)\n")
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        if bs > 64 and not USE_RTX8000:
            continue
        times = eval_folder(
            path_to_images,
            path_to_masks,
            output_dir,
            masker,
            paint,
            opts,
            bs,
            True,
            trainer,
            device,
            save_images,
            empty_cuda_cache,
            loaded_images,
        )
        print()

