from omnigan.trainer import Trainer
from omnigan.data import is_image_file
from omnigan.utils import Timer
from omnigan.tutils import normalize
import skimage.io as io
from skimage.transform import resize
import argparse
from pathlib import Path
import numpy as np


def print_time(text, time_series):
    """
    Print a timeseries's mean and std with a label

    Args:
        text (str): label of the time series
        time_series (list): list of timings
    """
    m = np.mean(time_series)
    s = np.std(time_series)
    print(f"{text.capitalize() + ' ':.<26}  {m:.5f} +/- {s:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
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
        "-t",
        "--time",
        action="store_true",
        default=False,
        help="Binary flag to time operations or not. Defaults to False.",
    )
    parser.add_argument(
        "-m",
        "--mask_binarization",
        type=float,
        default=-1,
        help="Value to use to binarize masks (mask > value). "
        + "Set to -1 (default) to use soft masks (not binarized)",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # ------------------------------------------------------
    # -----  Parse Arguments and initialize variables  -----
    # ------------------------------------------------------
    args = parse_args()
    print(
        "• Using args\n\n"
        + "\n".join(["{:15}: {}".format(k, v) for k, v in vars(args).items()]),
    )

    batch_size = args.batch_size
    images_paths = Path(args.images_paths).expanduser().resolve()
    mask_binarization = args.mask_binarization
    outdir = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else None
    )
    resume_path = args.resume_path
    time_inference = args.time

    if outdir is not None:
        outdir.mkdir(exist_ok=True, parents=True)

    # -------------------------------
    # -----  Create time store  -----
    # -------------------------------
    stores = {}
    if time_inference:
        stores = {
            "encode": [],
            "depth": [],
            "segmentation": [],
            "mask": [],
            "wildfire": [],
            "smog": [],
            "flood": [],
            "numpy": [],
            "setup": [],
            "inference on all images": [],
            "write": [],
        }

    # -------------------------------------
    # -----  Resume Trainer instance  -----
    # -------------------------------------
    print("\n• Initializing trainer\n")

    with Timer(store=stores.get("setup", [])):

        trainer = Trainer.resume_from_path(
            resume_path,
            setup=True,
            inference=True,
            new_exp=None,
            input_shapes=(3, 640, 640),
        )

    # --------------------------------------------
    # -----  Read data from input directory  -----
    # --------------------------------------------
    print("\n• Reading Data\n")

    # find all images
    data_paths = [i for i in images_paths.glob("*") if is_image_file(i)]
    # read images to numpy arrays
    data = [io.imread(str(d)) for d in data_paths]
    # resize to standard input size 640 x 640
    data = [resize(d, (640, 640), anti_aliasing=True) for d in data]
    # normalize to -1:1
    data = [(normalize(d.astype(np.float32)) - 0.5) * 2 for d in data]

    n_batchs = len(data) // args.batch_size + 1

    print("Found", len(data), "images.")

    # --------------------------------------------
    # -----  Batch-process images to events  -----
    # --------------------------------------------
    print(f"\n• Creating events on {str(trainer.device)}\n")

    all_events = []

    with Timer(store=stores.get("inference on all images", [])):
        for b in range(n_batchs):
            print(f"Batch {b + 1}/{n_batchs}", end="\r")
            images = data[b * batch_size : (b + 1) * batch_size]
            if not images:
                continue
            # concatenate images in a batch batch_size x height x width x 3
            images = np.stack(images)

            # Retreive numpy events as a dict {event: array}
            events = trainer.infer_all(
                images, True, stores, mask_binarization=mask_binarization
            )

            # store events to write after inference loop
            all_events.append(events)
    print()

    # ----------------------------------------------
    # -----  Write events to output directory  -----
    # ----------------------------------------------
    if outdir:
        print("\n• Writing")
        with Timer(store=stores.get("write", [])):
            for b, events in enumerate(all_events):
                for i in range(len(events)):
                    idx = b * batch_size + i
                    stem = Path(data_paths[idx]).stem
                    for event in events:
                        im_path = outdir / f"{stem}_{event}.png"
                        im_data = events[event][i]
                        io.imsave(im_path, im_data)

    # ---------------------------
    # -----  Print timings  -----
    # ---------------------------
    if time_inference:
        print("\n• Timings\n")
        for k in sorted(list(stores.keys())):
            print_time(k, stores[k])
