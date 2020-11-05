import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as trsfs

from omnigan.data import tensor_loader
from omnigan.trainer import Trainer
from omnigan.utils import load_opts

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


def print_time(name, time_series, precision=4, file=None):
    head = f"[{name}] Average time (per batch): "
    tail = ""
    if isinstance(time_series, (list, np.ndarray)):
        tail = (
            f"{np.mean(time_series):.{precision}f}s "
            + f"(+/- {np.std(time_series):.{precision}f}s)"
        )
    elif isinstance(time_series, str):
        tail = time_series

    print(head + tail)
    if file is not None:
        print(head + tail, file=file)


class Timer:
    def __init__(self, name="", store=None, precision=3):
        self.name = name
        self.store = store
        self.precision = precision

    def format(self, n):
        return f"{n:.{self.precision}f}"

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
            print(f"[{self.name}] Elapsed time: {self.format(new_time)}")


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


def prepare_image(
    img_numpy,
    new_size,
    transforms,
    device,
    use_half,
    to_tensor_time=[],
    transforms_time=[],
    to_device_time=[],
):
    with Timer(store=to_tensor_time):
        img_tensor = torch.from_numpy(img_numpy).unsqueeze(0)

    with Timer(store=transforms_time):
        img_tensor = F.interpolate(img_tensor, (new_size, new_size), mode="nearest")
        img_tensor = img_tensor.squeeze(0)
        for tf in transforms:
            img_tensor = tf(img_tensor)

        img_tensor = img_tensor.unsqueeze(0)
    with Timer(store=to_device_time):
        img_tensor = img_tensor.to(device)

    if use_half:
        img_tensor = img_tensor.half()

    return img_tensor


def prepare_mask(mask_tensor, device, use_half):
    mask_tensor = mask_tensor.squeeze().unsqueeze(0).to(device)
    if use_half:
        mask_tensor = mask_tensor.half()
    return mask_tensor


# @torch.no_grad() grad already disabled in __main__
def eval_folder(
    path_to_images,
    output_dir,
    batch_size,
    use_half,
    trainer,
    device,
    loaded_images,
    limit=-1,
    to_cpu=False,
    n_iter=1,
):
    model = trainer.G
    if use_half:
        model = model.half()
    model.eval()

    if not loaded_images:
        image_list = os.listdir(path_to_images)
        image_list.sort()
        images = [
            tensor_loader(path_to_images / Path(i), task="x", domain="val").numpy()[0]
            for i in image_list
        ]
    else:
        images = loaded_images
    if limit > 0:
        images = images[:limit]

    painter_inference_time = []
    masker_inference_time = []
    full_procedure_time = []
    inference_loop_time = []
    to_cpu_time = []
    to_tensor_time = []
    transforms_time = []
    to_device_time = []

    output_dir = output_dir / f"bs_{batch_size}{'_half' if use_half else ''}"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Batch Size:", batch_size)
    print("Using Half:", use_half)
    for it in range(n_iter):
        print(">>> Iteration {} for batch size {}\n".format(it, batch_size))
        with Timer(store=full_procedure_time):

            with Timer("Data Loading"):
                image_tensors = [
                    prepare_image(
                        im,
                        new_size,
                        transforms,
                        device,
                        use_half,
                        to_tensor_time,
                        transforms_time,
                        to_device_time,
                    )
                    for im in images
                ]

            with Timer(store=inference_loop_time):
                for i in range(len(image_tensors) // batch_size + 1):
                    img = image_tensors[i * batch_size : (i + 1) * batch_size]
                    if not img:
                        continue
                    img = torch.cat(img, axis=0)
                    print("Batch", i, img.shape, img.device, end="\r", flush=True)

                    with Timer(store=masker_inference_time):
                        z = model.encode(img)
                        mask = model.decoders["m"](z)
                        # xm.mark_step()

                    with Timer(store=painter_inference_time):
                        z_painter = None  # trainer.sample_z(1)
                        if use_half:
                            z_painter = z_painter.half()
                        fake_flooded = model.painter(z_painter, img * (1.0 - mask))
                        xm.mark_step()
                    if to_cpu:
                        with Timer(store=to_cpu_time):
                            fake_cpu = fake_flooded.cpu().numpy()

    batch_inference = np.array(masker_inference_time) + np.array(painter_inference_time)

    dump_first_n = 20
    batch_inference = batch_inference[dump_first_n:]
    masker_inference_time = masker_inference_time[dump_first_n:]
    painter_inference_time = painter_inference_time[dump_first_n:]

    with open(
        f"./eval_folder_nIm{len(images)}_bs{batch_size}_iter{n_iter}", "w"
    ) as write_file:

        print_time(
            "Full procedure (numpy->torch->transforms->device->infer) on"
            + f" {len(images)} images",
            full_procedure_time,
            file=write_file,
        )
        print_time("Inference loop (all dataset)", inference_loop_time, file=write_file)
        print_time("Single Batch (per batch)", batch_inference, file=write_file)
        print_time("Masker (per batch)", masker_inference_time, file=write_file)
        print_time("Painter (per batch)", painter_inference_time, file=write_file)
        print_time("To Tensor (per sample)", to_tensor_time, file=write_file)
        print_time("Transforms (per sample)", transforms_time, file=write_file)
        print_time("To Device (per sample)", to_device_time, file=write_file)
        print_time(
            "Back To CPU + Numpy (per batch)",
            to_cpu_time if to_cpu else "Not Measured",
            file=write_file,
        )

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--masker_dir", default="~/bucket/v1-weights/masker", type=str,
    )
    parser.add_argument(
        "-p", "--painter_dir", default="~/bucket/v1-weights/painter", type=str,
    )
    parser.add_argument(
        "-d", "--inference_data_dir", default="~/bucket/100postalcode", type=str,
    )
    parser.add_argument(
        "-o", "--output_dir", default="~/outputs", type=str,
    )
    parser.add_argument(
        "-c",
        "--to_cpu",
        default=False,
        action="store_true",
        help="Whether or not to count the time it takes "
        + "to move data from the device back to the cpu",
    )
    parser.add_argument(
        "-b",
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048],
        help="List of batch sizes to benchmark",
    )
    parser.add_argument(
        "-s",
        "--dataset_size",
        type=int,
        default=4096,
        help="Will repeat the images to match dataset_size",
    )
    parser.add_argument(
        "-n",
        "--n_iter",
        type=int,
        default=1,
        help="Benchmarking iterations per batch size",
    )

    args = parser.parse_args()
    print("Received Arguments:")
    print("\n".join(f"{k:20}: {v}" for k, v in vars(args).items()))

    # -----------------------
    # -----  Load opts  -----
    # -----------------------
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    masker_path = Path(args.masker_dir).expanduser().resolve()
    painter_path = Path(args.painter_dir).expanduser().resolve()

    assert masker_path.exists() and painter_path.exists()

    opts = load_opts(masker_path / "opts.yaml", default="shared/trainer/defaults.yaml")

    opts.tasks = ["m", "p"]
    opts.load_paths.m = str(masker_path)
    opts.load_paths.p = str(painter_path)
    opts.train.resume = True
    opts.output_path = output_dir
    opts.gen.p.latent_dim = 640

    new_size = 640

    paint = True
    masker = True

    # --------------------------------------
    # -----  Define trainer and model  -----
    # --------------------------------------
    torch.set_grad_enabled(False)
    device = xm.xla_device()
    trainer = Trainer(opts, device=device)
    trainer.input_shape = (3, 640, 640)
    trainer.setup(inference=True)
    trainer.resume(inference=True)

    # ------------------------
    # -----  Transforms  -----
    # ------------------------
    transforms = [trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # --------------------------------
    # -----  eval_folder params  -----
    # --------------------------------
    path_to_images = (
        Path(args.inference_data_dir).expanduser().resolve()
    )  # a folder with a list of images
    loaded_images = None  # will be overloaded with data if preload_images is True
    preload_images = True  # faster if running eval_folder multiple times
    limit = -1  # limit the number of images loaded, for debugging purposes
    to_cpu = args.to_cpu  # measure the time to bring tensors back to CPU from device
    dataset_size = args.dataset_size  # will repeat the 100 images to match this size
    batch_sizes = args.batch_sizes  # batch sizes to benchmark
    n_iter = args.n_iter

    # -----------------------------------
    # -----  Load images in memory  -----
    # -----------------------------------
    if preload_images:
        print("Pre-loading images in memory...", end="", flush=True)
        image_list = os.listdir(path_to_images)
        image_list.sort()
        loaded_images = [
            tensor_loader(path_to_images / Path(i), task="x", domain="val").numpy()[0]
            for i in image_list
        ]
        print(f"Total dataset size: {dataset_size}...", end="")
        loaded_images = loaded_images * (dataset_size // len(loaded_images) + 1)
        loaded_images = loaded_images[:dataset_size]
        print(" Ok.")

    # -----------------------
    # -----      -      -----
    # -----  Benchmark  -----
    # -----      -      -----
    # -----------------------

    for bs in batch_sizes:
        eval_folder(
            path_to_images,
            output_dir,
            bs,
            False,
            trainer,
            device,
            loaded_images,
            limit=limit,
            to_cpu=to_cpu,
            n_iter=n_iter,
        )
        print()
        with open(output_dir / f"omnigan_xla_metrics_bs{bs}_lim{limit}.txt", "w") as f:
            report = met.metrics_report()
            print(report, file=f)
