import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=4,
        help="Batch size to process input images to events. Defaults to 4",
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
        + "Will NOT write anything to disk if this flag is not used.",
    )
    parser.add_argument(
        "-s",
        "--save_input",
        action="store_true",
        default=False,
        help="Binary flag to include the input image to the model (after crop and"
        + " resize) in the images written or uploaded (depending on saving options.)",
    )
    parser.add_argument(
        "-r",
        "--resume_path",
        type=str,
        default=None,
        help="Path to a directory containing the trainer to resume."
        + " In particular it must contain `opts.yam` and `checkpoints/`."
        + " Typically this points to a Masker, which holds the path to a"
        + " Painter in its opts",
    )
    parser.add_argument(
        "--no_time",
        action="store_true",
        default=False,
        help="Binary flag to prevent the timing of operations.",
    )
    parser.add_argument(
        "-f",
        "--flood_mask_binarization",
        type=float,
        default=0.5,
        help="Value to use to binarize masks (mask > value). "
        + "Set to -1 to use soft masks (not binarized). Defaults to 0.5.",
    )
    parser.add_argument(
        "-t",
        "--target_size",
        type=int,
        default=640,
        help="Output image size (when not using `keep_ratio_128`): images are resized"
        + " such that their smallest side is `target_size` then cropped in the middle"
        + " of the largest side such that the resulting input image (and output images)"
        + " has height and width `target_size x target_size`. **Must** be a multiple of"
        + " 2^7=128 (up/downscaling inside the models). Defaults to 640.",
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
        help="Limit the number of images processed (if you have 100 images in "
        + "a directory but n is 10 then only the first 10 images will be loaded"
        + " for processing)",
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
        help="Do not check for existing outdir, i.e. force overwrite"
        + " potentially existing files in the output path",
    )
    parser.add_argument(
        "--no_cloudy",
        action="store_true",
        default=False,
        help="Prevent the use of the cloudy intermediate"
        + " image to create the flood image. Rendering will"
        + " be more colorful but may seem less realistic",
    )
    parser.add_argument(
        "--keep_ratio_128",
        action="store_true",
        default=False,
        help="When loading the input images, resize and crop them in order for their "
        + "dimensions to match the closest multiples"
        + " of 128. Will force a batch size of 1 since images"
        + " now have different dimensions. "
        + "Use --max_im_width to cap the resulting dimensions.",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        default=False,
        help="Use batch norm fusion to speed up inference",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=False,
        help="Save output masks along events",
    )
    parser.add_argument(
        "-m",
        "--max_im_width",
        type=int,
        default=-1,
        help="When using --keep_ratio_128, some images may still be too large. Use "
        + "--max_im_width to cap the resized image's width. Defaults to -1 (no cap).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to comet.ml in a project called `climategan-apply`",
    )
    parser.add_argument(
        "--zip_outdir",
        "-z",
        action="store_true",
        help="Zip the output directory as '{outdir.parent}/{outdir.name}.zip'",
    )
    return parser.parse_args()


args = parse_args()


print("\n• Imports\n")
import time

import_time = time.time()
import sys
import shutil
from collections import OrderedDict
from pathlib import Path

import comet_ml  # noqa: F401
import torch
import numpy as np
import skimage.io as io
from skimage.color import rgba2rgb
from skimage.transform import resize
from tqdm import tqdm

from climategan.trainer import Trainer
from climategan.bn_fusion import bn_fuse
from climategan.tutils import print_num_parameters
from climategan.utils import Timer, find_images, get_git_revision_hash, to_128, resolve

import_time = time.time() - import_time


def to_m1_p1(img, i):
    """
    rescales a [0, 1] image to [-1, +1]

    Args:
        img (np.array): float32 numpy array of an image in [0, 1]
        i (int): Index of the image being rescaled

    Raises:
        ValueError: If the image is not in [0, 1]

    Returns:
        np.array(np.float32): array in [-1, +1]
    """
    if img.min() >= 0 and img.max() <= 1:
        return (img.astype(np.float32) - 0.5) * 2
    raise ValueError(f"Data range mismatch for image {i} : ({img.min()}, {img.max()})")


def uint8(array):
    """
    convert an array to np.uint8 (does not rescale or anything else than changing dtype)

    Args:
        array (np.array): array to modify

    Returns:
        np.array(np.uint8): converted array
    """
    return array.astype(np.uint8)


def resize_and_crop(img, to=640):
    """
    Resizes an image so that it keeps the aspect ratio and the smallest dimensions
    is `to`, then crops this resized image in its center so that the output is `to x to`
    without aspect ratio distortion

    Args:
        img (np.array): np.uint8 255 image

    Returns:
        np.array: [0, 1] np.float32 image
    """
    # resize keeping aspect ratio: smallest dim is 640
    h, w = img.shape[:2]
    if h < w:
        size = (to, int(to * w / h))
    else:
        size = (int(to * h / w), to)

    r_img = resize(img, size, preserve_range=True, anti_aliasing=True)
    r_img = uint8(r_img)

    # crop in the center
    H, W = r_img.shape[:2]

    top = (H - to) // 2
    left = (W - to) // 2

    rc_img = r_img[top : top + to, left : left + to, :]

    return rc_img / 255.0


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
    """
    Saves the args to `apply_events.py` in a text file for future reference
    """
    cwd = Path.cwd().expanduser().resolve()
    command = f"cd {str(cwd)}\n"
    command += " ".join(sys.argv)
    git_hash = get_git_revision_hash()
    with (out / "command.txt").open("w") as f:
        f.write(command)
    with (out / "hash.txt").open("w") as f:
        f.write(git_hash)


def get_outdir_name(half, keep_ratio, max_im_width, target_size, bin_value, cloudy):
    """
    Create the output directory's name based on uer-provided arguments
    """
    name_items = []
    if half:
        name_items.append("half")
    if keep_ratio:
        name_items.append("AR")
    if max_im_width and keep_ratio:
        name_items.append(f"{max_im_width}")
    if target_size and not keep_ratio:
        name_items.append("S")
        name_items.append(f"{target_size}")
    if bin_value != 0.5:
        name_items.append(f"bin{bin_value}")
    if not cloudy:
        name_items.append("no_cloudy")

    return "-".join(name_items)


def make_outdir(
    outdir, overwrite, half, keep_ratio, max_im_width, target_size, bin_value, cloudy
):
    """
    Creates the output directory if it does not exist. If it does exist,
    prompts the user for confirmation (except if `overwrite` is True).
    If the output directory's name is "_auto_" then it is created as:
        outdir.parent / get_outdir_name(...)
    """
    if outdir.name == "_auto_":
        outdir = outdir.parent / get_outdir_name(
            half, keep_ratio, max_im_width, target_size, bin_value, cloudy
        )
    if outdir.exists() and not overwrite:
        print(
            f"\nWARNING: outdir ({str(outdir)}) already exists."
            + " Files with existing names will be overwritten"
        )
        if "n" in input(">>> Continue anyway? [y / n] (default: y) : "):
            print("Interrupting execution from user input.")
            sys.exit()
        print()
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


def get_time_stores(import_time):
    return OrderedDict(
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


if __name__ == "__main__":

    # -----------------------------------------
    # -----  Initialize script variables  -----
    # -----------------------------------------
    print(
        "• Using args\n\n"
        + "\n".join(["{:25}: {}".format(k, v) for k, v in vars(args).items()]),
    )

    batch_size = args.batch_size
    bin_value = args.flood_mask_binarization
    cloudy = not args.no_cloudy
    fuse = args.fuse
    half = args.half
    save_masks = args.save_masks
    images_paths = resolve(args.images_paths)
    keep_ratio = args.keep_ratio_128
    max_im_width = args.max_im_width
    n_images = args.n_images
    outdir = resolve(args.output_path) if args.output_path is not None else None
    resume_path = args.resume_path
    target_size = args.target_size
    time_inference = not args.no_time
    upload = args.upload
    zip_outdir = args.zip_outdir

    # -------------------------------------
    # -----  Validate size arguments  -----
    # -------------------------------------
    if keep_ratio:
        if target_size != 640:
            print(
                "\nWARNING: using --keep_ratio_128 overwrites target_size"
                + " which is ignored."
            )
        if batch_size != 1:
            print("\nWARNING: batch_size overwritten to 1 when using keep_ratio_128")
            batch_size = 1
        if max_im_width > 0 and max_im_width % 128 != 0:
            new_im_width = int(max_im_width / 128) * 128
            print("\nWARNING: max_im_width should be <0 or a multiple of 128.")
            print(
                "            Was {} but is now overwritten to {}".format(
                    max_im_width, new_im_width
                )
            )
            max_im_width = new_im_width
    else:
        if target_size % 128 != 0:
            print(f"\nWarning: target size {target_size} is not a multiple of 128.")
            target_size = target_size - (target_size % 128)
            print(f"Setting target_size to {target_size}.")

    # -------------------------------------
    # -----  Create output directory  -----
    # -------------------------------------
    if outdir is not None:
        outdir = make_outdir(
            outdir,
            args.overwrite,
            half,
            keep_ratio,
            max_im_width,
            target_size,
            bin_value,
            cloudy,
        )

    # -------------------------------
    # -----  Create time store  -----
    # -------------------------------
    stores = get_time_stores(import_time)

    # -----------------------------------
    # -----  Load Trainer instance  -----
    # -----------------------------------
    with Timer(store=stores.get("setup", []), ignore=time_inference):
        print("\n• Initializing trainer\n")
        torch.set_grad_enabled(False)
        trainer = Trainer.resume_from_path(
            resume_path,
            setup=True,
            inference=True,
            new_exp=None,
        )
        print()
        print_num_parameters(trainer, True)
        if fuse:
            trainer.G = bn_fuse(trainer.G)
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

    with Timer(store=stores.get("data pre-processing", []), ignore=time_inference):
        # read images to numpy arrays
        data = [io.imread(str(d)) for d in data_paths]
        # rgba to rgb
        data = [im if im.shape[-1] == 3 else uint8(rgba2rgb(im) * 255) for im in data]
        # resize images to target_size or
        if keep_ratio:
            # to closest multiples of 128 <= max_im_width, keeping aspect ratio
            new_sizes = [to_128(d, max_im_width) for d in data]
            data = [resize(d, ns, anti_aliasing=True) for d, ns in zip(data, new_sizes)]
        else:
            # to args.target_size
            data = [resize_and_crop(d, target_size) for d in data]
            new_sizes = [(target_size, target_size) for _ in data]
        # resize() produces [0, 1] images, rescale to [-1, 1]
        data = [to_m1_p1(d, i) for i, d in enumerate(data)]

    n_batchs = len(data) // batch_size
    if len(data) % batch_size != 0:
        n_batchs += 1

    print("Found", len(base_data_paths), "images. Inferring on", len(data), "images.")

    # --------------------------------------------
    # -----  Batch-process images to events  -----
    # --------------------------------------------
    print(f"\n• Using device {str(trainer.device)}\n")

    all_events = []

    with Timer(store=stores.get("inference on all images", []), ignore=time_inference):
        for b in tqdm(range(n_batchs), desc="Infering events", unit="batch"):

            images = data[b * batch_size : (b + 1) * batch_size]
            if not images:
                continue

            # concatenate images in a batch batch_size x height x width x 3
            images = np.stack(images)
            # Retreive numpy events as a dict {event: array[BxHxWxC]}
            events = trainer.infer_all(
                images,
                numpy=True,
                stores=stores,
                bin_value=bin_value,
                half=half,
                cloudy=cloudy,
                return_masks=save_masks,
            )

            # save resized and cropped image
            if args.save_input:
                events["input"] = uint8((images + 1) / 2 * 255)

            # store events to write after inference loop
            all_events.append(events)

    # --------------------------------------------
    # -----  Save (write/upload) inferences  -----
    # --------------------------------------------
    if outdir is not None or upload:

        if upload:
            print("\n• Creating comet Experiment")
            exp = comet_ml.Experiment(project_name="climategan-apply")
            exp.log_parameters(vars(args))

        # --------------------------------------------------------------
        # -----  Change inferred data structure to a list of dicts  -----
        # --------------------------------------------------------------
        to_write = []
        events_names = list(all_events[0].keys())
        for events_data in all_events:
            n_ims = len(events_data[events_names[0]])
            for i in range(n_ims):
                item = {event: events_data[event][i] for event in events_names}
                to_write.append(item)

        progress_bar_desc = ""
        if outdir is not None:
            print("\n• Output directory:\n")
            print(str(outdir), "\n")
            if upload:
                progress_bar_desc = "Writing & Uploading events"
            else:
                progress_bar_desc = "Writing events"
        else:
            if upload:
                progress_bar_desc = "Uploading events"

        # ------------------------------------
        # -----  Save individual images  -----
        # ------------------------------------
        with Timer(store=stores.get("write", []), ignore=time_inference):

            # for each image
            for t, event_dict in tqdm(
                enumerate(to_write),
                desc=progress_bar_desc,
                unit="input image",
                total=len(to_write),
            ):

                idx = t % len(base_data_paths)
                stem = Path(data_paths[idx]).stem
                width = new_sizes[idx][1]

                if keep_ratio:
                    ar = "_AR"
                else:
                    ar = ""

                # for each event type
                event_bar = tqdm(
                    enumerate(event_dict.items()),
                    leave=False,
                    total=len(events_names),
                    unit="event",
                )
                for e, (event, im_data) in event_bar:
                    event_bar.set_description(
                        f"  {event.capitalize():<{len(progress_bar_desc) - 2}}"
                    )

                    if args.no_cloudy:
                        suffix = ar + "_no_cloudy"
                    else:
                        suffix = ar

                    im_path = Path(f"{stem}_{event}_{width}{suffix}.png")

                    if outdir is not None:
                        im_path = outdir / im_path
                        io.imsave(im_path, im_data)

                    if upload:
                        exp.log_image(im_data, name=im_path.name)
    if zip_outdir:
        print("\n• Zipping output directory... ", end="", flush=True)
        archive_path = Path(shutil.make_archive(outdir.name, "zip", root_dir=outdir))
        archive_path = archive_path.rename(outdir.parent / archive_path.name)
        print("Done:\n")
        print(str(archive_path))

    # ---------------------------
    # -----  Print timings  -----
    # ---------------------------
    if time_inference:
        print("\n• Timings\n")
        print_store(stores)

    # ---------------------------------------------
    # -----  Save apply_events.py run config  -----
    # ---------------------------------------------
    if not args.no_conf and outdir is not None:
        write_apply_config(outdir)
