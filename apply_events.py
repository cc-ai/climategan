print("\n• Imports\n")
import time

import_time = time.time()
import sys
import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import skimage.io as io
from skimage.color import rgba2rgb
from skimage.transform import resize
import comet_ml  # noqa: F401

from omnigan.trainer import Trainer
from omnigan.data import get_inference_loader
from omnigan.tutils import normalize, print_num_parameters
from omnigan.utils import (
    Timer,
    get_git_revision_hash,
    to_128,
    find_images,
)

import_time = time.time() - import_time

XLA = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    XLA = True
except ImportError:
    pass


def print_time(text, time_series, purge=-1):
    """
    Print a timeseries's mean and std with a label

    Args:
        text (str): label of the time series
        time_series (list): list of timings
        purge (int, optional): ignore first n values of time series. Defaults to -1.
    """
    if not time_series:
        return

    if purge > 0 and len(time_series) > purge:
        time_series = time_series[purge:]

    m = np.mean(time_series)
    s = np.std(time_series)

    print(
        f"{text.capitalize() + ' ':.<26}  {m:.5f}"
        + (f" +/- {s:.5f}" if len(time_series) > 1 else "")
    )


def print_store(store, purge=-1):
    """
    Pretty-print time series store

    Args:
        store (dict): maps string keys to lists of times
        purge (int, optional): ignore first n values of time series. Defaults to -1.
    """
    singles = OrderedDict({k: v for k, v in store.items() if len(v) == 1})
    multiples = OrderedDict({k: v for k, v in store.items() if len(v) > 1})
    empties = {k: v for k, v in store.items() if len(v) == 0}

    if empties:
        print("Ignoring empty stores ", ", ".join(empties.keys()))
        print()

    for k in singles:
        print_time(k, singles[k], purge)

    print()
    print("Unit: s/batch")
    for k in multiples:
        print_time(k, multiples[k], purge)
    print()


def write_apply_config(out):
    command = " ".join(sys.argv)
    git_hash = get_git_revision_hash()
    with (out / "command.txt").open("w") as f:
        f.write(command)
    with (out / "hash.txt").open("w") as f:
        f.write(git_hash)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=4,
        help="Batch size to process input images to events",
    )
    parser.add_argument(
        "-i",
        "--images_paths",
        type=str,
        required=True,
        help="Path to a directory with image files",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to a directory were events should be written. "
        + "Will NOT write anything if this flag is not used.",
    )
    parser.add_argument(
        "-r",
        "--resume_path",
        type=str,
        default=None,
        help="Path to a directory containing the trainer to resume."
        + " In particular it must contain opts.yaml and checkpoints/",
    )
    parser.add_argument(
        "--no_time",
        action="store_true",
        default=False,
        help="Binary flag to prevent the timing of operations.",
    )
    parser.add_argument(
        "-m",
        "--flood_mask_binarization",
        type=float,
        default=0.5,
        help="Value to use to binarize masks (mask > value). "
        + "Set to -1 (default) to use soft masks (not binarized)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Binary flag to use half precision (float16). Defaults to False.",
    )
    parser.add_argument(
        "-n",
        "--n_images",
        default=-1,
        type=int,
        help="Limit the number of images processed",
    )
    parser.add_argument(
        "-x",
        "--xla_purge_samples",
        type=int,
        default=-1,
        help="XLA compile time induces extra computations."
        + " Ignore -x samples when computing time averages",
    )
    parser.add_argument(
        "--no_conf",
        action="store_true",
        default=False,
        help="disable writing the apply_events hash and command in the output folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Do not check for existing outdir",
    )
    parser.add_argument(
        "--no_cloudy",
        action="store_true",
        default=False,
        help="Prevent the use of the cloudy intermediate"
        + " image to create the flood image",
    )
    parser.add_argument(
        "--keep_ratio_128",
        action="store_true",
        default=False,
        help="Keeps approximately the aspect ratio to match multiples of 128",
    )
    parser.add_argument(
        "--data_loader",
        action="store_true",
        default=False,
        help="Use a DataLoader to load data",
    )
    parser.add_argument(
        "--max_im_width",
        type=int,
        default=-1,
        help="Maximum image width: will downsample larger images",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # ------------------------------------------------------
    # -----  Parse Arguments and initialize variables  -----
    # ------------------------------------------------------
    args = parse_args()
    print(
        "• Using args\n\n"
        + "\n".join(["{:25}: {}".format(k, v) for k, v in vars(args).items()]),
    )

    batch_size = args.batch_size
    half = args.half
    cloudy = not args.no_cloudy
    images_paths = Path(args.images_paths).expanduser().resolve()
    bin_value = args.flood_mask_binarization
    loader = data_iter = data = None
    outdir = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else None
    )
    resume_path = args.resume_path
    time_inference = not args.no_time
    n_images = args.n_images
    xla_purge_samples = args.xla_purge_samples
    if args.keep_ratio_128:
        if batch_size != 1:
            print("\nWARNING: batch_size overwritten to 1 when using keep_ratio_128")
            batch_size = 1
        if args.max_im_width > 0 and args.max_im_width % 128 != 0:
            new_im_width = int(args.max_im_width / 128) * 128
            print("\nWARNING: max_im_width should be <0 or a multiple of 128.")
            print(
                "            Was {} but is now overwritten to {}".format(
                    args.max_im_width, new_im_width
                )
            )
            args.max_im_width = new_im_width

    if outdir is not None:
        if outdir.exists() and not args.overwrite:
            print(
                f"\nWARNING: outdir ({str(outdir)}) already exists."
                + " Files with existing names will be overwritten"
            )
            if "n" in input(">>> Continue anyway? [y / n] (default: y) : "):
                print("Interrupting execution from user input.")
                sys.exit()
            print()
        outdir.mkdir(exist_ok=True, parents=True)

    # -------------------------------
    # -----  Create time store  -----
    # -------------------------------
    stores = {}
    if time_inference:
        stores = OrderedDict(
            {
                "imports": [import_time],
                "setup": [],
                "data pre-processing": [],
                "encode": [],
                "mask": [],
                "flood": [],
                "depth": [],
                "segmentation": [],
                "smog": [],
                "wildfire": [],
                "all events": [],
                "numpy": [],
                "inference on all images": [],
                "write": [],
            }
        )

    # -------------------------------------
    # -----  Resume Trainer instance  -----
    # -------------------------------------
    print("\n• Initializing trainer\n")

    with Timer(store=stores.get("setup", [])):

        device = None
        if XLA:
            device = xm.xla_device()

        trainer = Trainer.resume_from_path(
            resume_path, setup=True, inference=True, new_exp=None, device=device,
        )
        print()
        print_num_parameters(trainer, True)
        if half:
            trainer.G.half()

    # --------------------------------------------
    # -----  Read data from input directory  -----
    # --------------------------------------------
    print("\n• Reading & Pre-processing Data\n")

    # find all images
    data_paths = find_images(images_paths)
    base_data_paths = data_paths
    # filter images
    if 0 < n_images < len(data_paths):
        data_paths = data_paths[:n_images]
    # repeat data
    elif n_images > len(data_paths):
        repeats = n_images // len(data_paths) + 1
        data_paths = base_data_paths * repeats
        data_paths = data_paths[:n_images]

    with Timer(store=stores.get("data pre-processing", [])):
        if not args.data_loader:
            # read images to numpy arrays
            data = [io.imread(str(d)) for d in data_paths]
            # rgba to rgb
            data = [im if im.shape[-1] == 3 else rgba2rgb(im) for im in data]
            # resize to standard input size 640 x 640
            if args.keep_ratio_128:
                new_sizes = [to_128(d, args.max_im_width) for d in data]
                data = [
                    resize(d, ns, anti_aliasing=True) for d, ns in zip(data, new_sizes)
                ]
            else:
                data = [resize(d, (640, 640), anti_aliasing=True) for d in data]
            # normalize to -1:1
            data = [(normalize(d.astype(np.float32)) - 0.5) * 2 for d in data]
            n_batchs = len(data) // batch_size
            if len(data) % batch_size != 0:
                n_batchs += 1
        else:
            loader = get_inference_loader(images_paths, batch_size, 640, args.half, 4)
            data_iter = iter(loader)
            n_batchs = len(loader.dataset) // batch_size
            if len(loader.dataset) % batch_size != 0:
                n_batchs += 1

    print("Found", len(base_data_paths), "images. Inferring on", len(data), "images.")

    # --------------------------------------------
    # -----  Batch-process images to events  -----
    # --------------------------------------------
    print(f"\n• Creating events on {str(trainer.device)}\n")

    all_events = []

    with Timer(store=stores.get("inference on all images", [])):
        for b in range(n_batchs):
            print(f"Batch {b + 1}/{n_batchs}", end="\r")
            images = (
                data[b * batch_size : (b + 1) * batch_size]
                if loader is None
                else next(data_iter)
            )
            if not images:
                continue
            # concatenate images in a batch batch_size x height x width x 3
            if isinstance(images, list):
                images = np.stack(images)

            # Retreive numpy events as a dict {event: array}
            events = trainer.infer_all(
                images,
                numpy=True,
                stores=stores,
                bin_value=bin_value,
                half=half,
                xla=XLA,
                cloudy=cloudy,
            )

            # store events to write after inference loop
            all_events.append(events)
    print()

    # ----------------------------------------------
    # -----  Write events to output directory  -----
    # ----------------------------------------------
    if outdir is not None:
        print("\n• Writing")
        n_written = 0
        with Timer(store=stores.get("write", [])):
            for b, events in enumerate(all_events):
                for i in range(len(list(events.values())[0])):

                    print(" " * 30, end="\r", flush=True)
                    print(
                        f"{n_written+1}/{len(base_data_paths)} ...",
                        end="\r",
                        flush=True,
                    )

                    idx = b * batch_size + i
                    idx = idx % len(base_data_paths)
                    stem = Path(data_paths[idx]).stem

                    for event in events:
                        im_path = outdir / f"{stem}_{event}.png"
                        im_data = events[event][i]
                        io.imsave(im_path, im_data)

                    n_written += 1

    # ---------------------------
    # -----  Print timings  -----
    # ---------------------------
    if time_inference:
        print("\n• Timings\n")
        print_store(stores, purge=xla_purge_samples)

    if XLA:
        metrics_dir = Path(__file__).parent / "config" / "metrics"
        metrics_dir.mkdir(exist_ok=True, parents=True)
        now = str(datetime.now()).replace(" ", "_")
        with open(metrics_dir / f"xla_metrics_{now}.txt", "w",) as f:
            report = met.metrics_report()
            print(report, file=f)

    if not args.no_conf and outdir is not None:
        write_apply_config(outdir)
