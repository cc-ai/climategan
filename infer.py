import time


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        """Start a new timer as a context manager"""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        s = "\n"
        t = time.perf_counter()
        if self.name:
            s += f"[{self.name}] "
        print(s + f"Elapsed time: {t - self._start_time:.3f}\n")


with Timer("Imports"):
    from torch.utils.data import Dataset, DataLoader
    from omnigan.utils import load_opts
    from pathlib import Path
    from argparse import ArgumentParser
    from omnigan.trainer import Trainer
    from omnigan.data import tensor_loader
    from torchvision.transforms import Normalize
    import torchvision.utils as vutils
    import torch.nn.functional as F
    import os
    from tqdm import tqdm


TRANSFORMS = [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


class InferDataset(Dataset):
    def __init__(self, path, output_size=640):
        self.path = Path(path)
        self.paths = [str(p.resolve()) for p in self.path.glob("*") if isimg(p)]
        self.output_size = output_size

    def load(self, path):
        img = tensor_loader(path, task="x", domain="val")
        img = F.interpolate(img, (self.output_size, self.output_size), mode="nearest")
        img = img.squeeze(0)
        for tf in TRANSFORMS:
            img = tf(img)
        return img

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return {"x": self.load(self.paths[index]), "path": self.paths[index]}


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
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size for inference. "
        + "Set to -1 to disable (infer 1 by 1, no DataLoader)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers for the DataLoader"
        + "(only relevant if batch_size > 0)",
    )
    parser.add_argument(
        "--keep_in_memory",
        default=False,
        action="store_true",
        help="Without this flag, images are written after each batch. "
        + "With the flag they are kept in memory and written after all inferences "
        + "are performed (only relevant if batch_size > 0)",
    )
    parser.add_argument(
        "--no_inference_mode",
        default=False,
        action="store_true",
        help="Initialize the trainer without the inference mode",
    )

    return parser.parse_args()


def eval_folder(path_to_images, output_dir, paint=False, device="cpu"):
    images = [path_to_images / Path(i) for i in os.listdir(path_to_images)]
    for img_path in images:
        img = tensor_loader(img_path, task="x", domain="val")
        # Resize img:
        img = F.interpolate(img, (new_size, new_size), mode="nearest")
        img = img.squeeze(0)
        for tf in TRANSFORMS:
            img = tf(img)

        img = img.unsqueeze(0).to(device)
        z = model.encode(img)
        mask = model.decoders["m"](z)

        vutils.save_image(mask, output_dir / ("mask_" + img_path.name), normalize=True)

        if paint:
            z_painter = trainer.sample_z(1)
            fake_flooded = model.painter(z_painter, img * (1.0 - mask))
            vutils.save_image(fake_flooded, output_dir / img_path.name, normalize=True)


def batch_eval_folder(
    path_to_images,
    outputdir,
    model,
    output_size=640,
    batch_size=8,
    num_workers=8,
    paint=False,
    keep_in_memory=True,
    device="cpu",
):
    dataset = InferDataset(path_to_images, output_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    masks = []
    painted = []
    paths = []

    for img in tqdm(dataloader, desc="Inferring"):
        z = model.encode(img["x"].to(device))
        mask = model.decoders["m"](z)

        if keep_in_memory:
            masks.extend(list(mask.detach().cpu().numpy()))
            paths.extend(img["path"])
        else:
            for k, m in enumerate(mask):
                vutils.save_image(
                    m,
                    output_dir / ("mask_" + Path(img["path"][k]).name),
                    normalize=True,
                )

        if paint:
            z_painter = trainer.sample_z(img["x"].shape[0])
            fake_flooded = model.painter(z_painter, img * (1.0 - mask))
            if keep_in_memory:
                painted.extend(list(fake_flooded.detach().cpu().numpy()))
            else:
                for k, fake in enumerate(fake_flooded):
                    vutils.save_image(
                        fake, output_dir / Path(img["path"][k]).name, normalize=True
                    )

    if keep_in_memory:
        for mask, fake, path in tqdm(
            zip(masks, painted, paths), total=len(masks), desc="Saving Images"
        ):
            vutils.save_image(fake, output_dir / Path(path).name, normalize=True)
            vutils.save_image(
                mask, output_dir / ("mask_" + Path(path).name), normalize=True,
            )


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

    new_size = None
    if args.new_size is None:
        for tf in opts.data.transforms:
            if tf["name"] == "resize":
                new_size = tf["new_size"]
    else:
        new_size = args.new_size

    if new_size is None:
        print("Warning: no size provided, defaulting to 640px")
        new_size = 640

    if "m" in opts.tasks and "p" in opts.tasks:
        paint = True
    else:
        paint = False
    # ------------------------
    # ----- Define model -----
    # ------------------------
    inference = not args.no_inference_mode
    trainer = Trainer(opts)
    trainer.input_shape = (3, 640, 640)
    with Timer("trainer.setup"):
        trainer.setup(inference=inference)
    with Timer("trainer.resume"):
        trainer.resume(inference=inference)
    model = trainer.G
    model.eval()

    # -----------------------------------------------
    # -----  Iterate Subdirs in Base Directory  -----
    # -----------------------------------------------

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
            # -------------------------
            # -----  Eval Folder  -----
            # -------------------------
            if args.batch_size <= 0:
                with Timer("eval_folder"):
                    eval_folder(root, write_path, paint, trainer.device)
            else:
                with Timer("batch_eval_folder"):
                    batch_eval_folder(
                        root,
                        write_path,
                        model,
                        new_size,
                        args.batch_size,
                        args.num_workers,
                        paint,
                        args.keep_in_memory,
                        trainer.device,
                    )

