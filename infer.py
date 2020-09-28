import torch
import numpy
from omnigan.utils import load_opts
from pathlib import Path
from argparse import ArgumentParser
from omnigan.trainer import Trainer
from omnigan.data import tensor_loader
from torchvision import transforms as trsfs
import torchvision.utils as vutils
import torch.nn.functional as F
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
        "--new_size",
        type=int,
        help="Size of generated masks",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/",
        type=str,
        help="Directory to write images to",
    )
    parser.add_argument(
        "--path_to_masks",
        type=str,
        help="Path of masks to be used for painting",
        required=False,
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        help="Apply mask to image to save",
    )

    return parser.parse_args()


def eval_folder(
    output_dir,
    path_to_images,
    path_to_masks=None,
    paint=False,
    masker=False,
    apply_mask=False,
):

    image_list = os.listdir(path_to_images)
    image_list.sort()
    images = [path_to_images / Path(i) for i in image_list]
    if not masker:
        mask_list = os.listdir(path_to_masks)
        mask_list.sort()
        masks = [path_to_masks / Path(i) for i in mask_list]

    for i, img_path in enumerate(images):
        img = tensor_loader(img_path, task="x", domain="val")

        # Resize img:
        img = F.interpolate(img, (new_size, new_size), mode="nearest")
        img = img.squeeze(0)
        for tf in transforms:
            img = tf(img)

        img = img.unsqueeze(0).to(device)

        if not masker:
            mask = tensor_loader(masks[i], task="m", domain="val")
            mask = F.interpolate(mask, (new_size, new_size), mode="nearest")
            mask = mask.squeeze()
            mask = mask.unsqueeze(0).to(device)

        if masker:
            if "m2" in opts.tasks:
                z = model.encode(img)
                z_aug_1 = torch.cat(
                    (z, trainer.label_1[0, :, :, :].unsqueeze(0)),
                    dim=1,
                )
                z_aug_2 = torch.cat(
                    (z, trainer.label_2[0, :, :, :].unsqueeze(0)),
                    dim=1,
                )
                mask_1 = model.decoders["m"](z_aug_1)
                mask_2 = model.decoders["m"](z_aug_2)
                vutils.save_image(
                    mask_1, output_dir / ("mask1_" + img_path.name), normalize=True
                )
                vutils.save_image(
                    mask_2, output_dir / ("mask2_" + img_path.name), normalize=True
                )

                if apply_mask:
                    vutils.save_image(
                        img * (1.0 - mask_1) + mask_1,
                        output_dir / (img_path.stem + "img_masked_1" + ".jpg"),
                        normalize=True,
                    )
                    vutils.save_image(
                        img * (1.0 - mask_2) + mask_2,
                        output_dir / (img_path.stem + "img_masked_2" + ".jpg"),
                        normalize=True,
                    )

            else:
                z = model.encode(img)
                mask = model.decoders["m"](z)
                vutils.save_image(
                    mask, output_dir / ("mask_" + img_path.name), normalize=True
                )

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

    paint = False
    masker = False
    if "p" in opts.tasks:
        paint = True
    if "m" in opts.tasks:
        masker = True
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

    transforms = [trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------

    # eval_folder(args.path_to_images, output_dir)

    rootdir = args.path_to_images
    maskdir = args.path_to_masks
    writedir = args.output_dir

    for root, subdirs, files in tqdm(os.walk(rootdir)):
        root = Path(root)
        subdirs = [Path(subdir) for subdir in subdirs]
        files = [Path(f) for f in files]
        has_imgs = False
        for f in files:
            if isimg(f):
                has_imgs = True
                break

        if has_imgs:
            print(f"Eval on {root}")
            rel_path = root.relative_to(rootdir)
            write_path = writedir / rel_path
            write_path.mkdir(parents=True, exist_ok=True)
            print("root: ", root)
            eval_folder(
                write_path,
                root,
                path_to_masks=maskdir,
                paint=paint,
                masker=masker,
                apply_mask=args.apply_mask,
            )
