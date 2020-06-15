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
        default="./output_masks/",
        type=str,
        help="Directory to write images to",
    )

    return parser.parse_args()


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
        trsfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------

    images = [args.path_to_images / Path(i) for i in os.listdir(args.path_to_images)]
    for img_path in images:
        img = pil_image_loader(img_path, task="x")
        # Resize img:
        img = TF.resize(img, (new_size, new_size))

        for tf in transforms:
            img = tf(img)

        img = img.unsqueeze(0).to(device)
        z = model.encode(img)
        mask = model.decoders["m"](z)
        vutils.save_image(mask, output_dir / img_path.name)
