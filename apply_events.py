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
    m = np.mean(time_series)
    s = np.std(time_series)
    print(f"  {text.capitalize():20} -> {m:.5f} +/- {s:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-i", "--images_paths", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("-r", "--resume_path", type=str, default=None)
    parser.add_argument("-t", "--time", action="store_true", default=False)
    parser.add_argument("-m", "--mask_binarization", type=float, default=-1)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print("Received", vars(args))

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
            "inference": [],
            "write": [],
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
    data = [
        resize(io.imread(str(i)), (640, 640), anti_aliasing=True) for i in data_paths
    ]
    data = [(normalize(i.astype(np.float32)) - 0.5) * 2 for i in data]

    n_batchs = len(data) // args.batch_size + 1

    print("\n• Creating events\n")

    all_events = []

    with Timer(store=stores.get("inference", [])):
        for b in range(n_batchs):
            print(f"Batch {b + 1}/{n_batchs}...", end="\r")
            images = data[b * batch_size : (b + 1) * batch_size]
            if not images:
                continue

            images = np.stack(images)

            events = trainer.infer_all(
                images, True, stores, mask_binarization=mask_binarization
            )
            all_events.append(events)

    if outdir:
        with Timer(store=stores.get("write", [])):
            for b, events in enumerate(all_events):
                for i in range(len(events)):
                    idx = b * batch_size + i
                    stem = Path(data_paths[idx]).stem
                    for event in events:
                        im_path = outdir / f"{stem}_{event}.png"
                        im_data = events[event][i]
                        io.imsave(im_path, im_data)

    if time_inference:
        print("\n• Timings:")
        for k in sorted(list(stores.keys())):
            print_time(k, stores[k])
