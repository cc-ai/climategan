from omnigan.trainer import Trainer
from omnigan.data import is_image_file
from omnigan.utils import Timer
import skimage.io as io
import argparse
from pathlib import Path
import numpy as np


def print_time(text, time_series):
    m = np.mean(time_series)
    s = np.std(time_series)
    print(f"  {text.capitalize()} -> {m} +/- {s}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-i", "--images_paths", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("-r", "--resume_path", type=str, default=None)
    parser.add_argument("-t", "--time", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print("Received", vars(args))

    batch_size = args.batch_size
    images_paths = Path(args.images_paths).expanduser().resolve()
    outdir = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else None
    )
    resume_path = args.resume_path
    time_inference = args.time

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
        }

    outdir.mkdir(exist_ok=True, parents=True)

    print("\n• Initializing trainer\n")

    with Timer(store=stores.get("setup", [])):

        trainer = Trainer.resume_from_path(
            resume_path,
            setup=True,
            inference=True,
            new_exp=None,
            input_shapes=(3, 640, 640),
        )

    print("\n• Reading Data\n")

    data_paths = [i for i in images_paths.glob("*") if is_image_file(i)]
    data = [io.imread(str(i)) for i in data_paths]

    n_batchs = len(data) // args.batch_size + 1

    print("\n• Creating events\n")

    for b in range(n_batchs):
        print(f"Batch {b + 1}/{n_batchs}...", end="\r")
        images = data[b * batch_size : (b + 1) * batch_size]
        if not images:
            continue

        images = np.vstack(images)

        events = trainer.infer_all(images, True, stores)

        if outdir:
            for i in range(len(images)):
                idx = b * batch_size + i
                stem = Path(data_paths[idx]).stem
                for event in events:
                    im_path = outdir / f"{stem}_{event}.png"
                    im_data = events[event][idx]
                    io.imsave(im_path, im_data)

    if time_inference:
        print("\n• Timings:")
        for k, v in stores.items():
            print_time(k, v)
