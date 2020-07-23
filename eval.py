import torch
import numpy
from omnigan.utils import load_opts
from pathlib import Path
from argparse import ArgumentParser
from omnigan.trainer import Trainer
from omnigan.data import pil_image_loader
from torchvision import transforms as trsfs
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import os
from tqdm import tqdm


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./shared/trainer/defaults.yaml",
        type=str,
        help="What configuration file to use to overwrite default",
    )
    parser.add_argument(
        "--default_config",
        default="./shared/trainer/defaults.yaml",
        type=str,
        help="What default file to use",
    )
    parser.add_argument(
        "--path_to_images",
        type=str,
        help="Path of images to be inferred",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to experiment folder containing checkpoints/latest_ckpt.pth",
        required=True,
    )
    parser.add_argument(
        "--new_size", type=int, help="Size of generated masks",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/",
        type=str,
        help="Directory to write images to",
    )

    return parser.parse_args()


def eval_folder(path_to_images, output_dir, paint=False):
    images = [path_to_images / Path(i) for i in os.listdir(path_to_images)]
    for img_path in images:
        img = pil_image_loader(img_path, task="x", domain="val")
        # Resize img:
        img = TF.resize(img, (new_size, new_size))
        for tf in transforms:
            img = tf(img)

        img = img.unsqueeze(0).to(device)
        z = model.encode(img)
        mask = model.decoders["m"](z)

        vutils.save_image(mask, output_dir / ("mask_" + img_path.name), normalize=True)

        if paint:
            z_painter = trainer.sample_z(1)
            fake_flooded = model.painter(z_painter, img * (1.0 - mask))
            vutils.save_image(fake_flooded, output_dir / img_path.name, normalize=True)


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


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(Path(args.config), default="./shared/trainer/defaults.yaml")
    opts.train.resume = True
    opts.output_path = str(Path(args.checkpoint).resolve())
    if args.new_size is None:
        for tf in opts.data.transforms:
            if tf["name"] == "resize":
                new_size = tf["new_size"]
    else:
        new_size = args.new_size

    if "m" in opts.tasks and "p" in opts.tasks:
        paint = True
    else:
        paint = False
    # ------------------------
    # ----- Define model -----
    # ------------------------

    trainer = Trainer(opts)
    trainer.setup()
    trainer.resume()
    model = trainer.G
    model.eval()

    # -------------------------------
    # -----  Transforms images  -----
    # -------------------------------

    transforms = [
        trsfs.ToTensor(),
        trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------

    # eval_folder(args.path_to_images, output_dir)

    rootdir = args.path_to_images
    writedir = args.output_dir

    for root, subdirs, files in tqdm(os.walk(rootdir)):
        root = Path(root)
        subdirs = [Path(subdir) for subdir in subdirs]
        files = [Path(f) for f in files]
        has_imgs = False
        for f in files:
            if isimg(f):
                # read_path = root / f
                # rel_path = read_path.relative_to(rootdir)
                # write_path = writedir / rel_path
                # write_path.mkdir(parents=True, exist_ok=True)
                has_imgs = True
                break

        if has_imgs:
            print(f"Eval on {root}")
            rel_path = root.relative_to(rootdir)
            write_path = writedir / rel_path
            write_path.mkdir(parents=True, exist_ok=True)
            eval_folder(root, write_path, paint)
