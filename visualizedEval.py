from comet_ml import Experiment
import torch
import numpy
from omnigan.utils import load_opts, flatten_opts
from pathlib import Path
from argparse import ArgumentParser
from omnigan.trainer import Trainer
from omnigan.data import get_all_loaders, pil_image_loader
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
    parser.add_argument(
        "--image_domain",
        default="r",
        type=str,
        help="Domain of images in path_to_images, can be 'r' or 's'",
        required=True,
    )
    parser.add_argument(
        "--no_comet", action="store_true", help="DON'T use comet.ml to log experiment"
    )
    return parser.parse_args()


def eval_folder(trainer, opts, args):
    save_images = {}
    update_task = "m"

    for i, multi_batch_tuple in enumerate(trainer.val_loaders):
        print("num_tuple: ", i)
        multi_domain_batch = {
            batch["domain"][0]: trainer.batch_to_device(batch)
            for batch in multi_batch_tuple
        }
        save_images[update_task] = []
        j = 0
        for batch_domain, batch in multi_domain_batch.items():
            print(j)
            j += 1
            if batch_domain == args.image_domain:
                name = batch["paths"]["m"][0].split("/")[-1]
                x = batch["data"]["x"]
                m = batch["data"]["m"]
                z = model.encode(x)
                mask = model.decoders[update_task](z)
                mask = mask.repeat(1, 3, 1, 1)
                task_saves = []

                task_saves.append(x * (1.0 - mask))
                task_saves.append(x * (1.0 - m.repeat(1, 3, 1, 1)))
                task_saves.append(mask)
                save_images[update_task].append(x)
                for im in task_saves:
                    save_images[update_task].append(im)
                vutils.save_image(mask, output_dir / name)
        write_images(
            trainer,
            image_outputs=save_images[update_task],
            mode="val",
            domain=args.image_domain,
            task=update_task,
            im_per_row=opts.comet.im_per_row.get(update_task, 4),
            comet_exp=exp,
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


def write_images(
    trainer, image_outputs, mode, domain, task, im_per_row=3, comet_exp=None
):
    """Save output image
    Arguments:
        image_outputs {Tensor list} -- list of output images
        im_per_row {int} -- number of images to be displayed (per row)
        file_name {str} -- name of the file where to save the images
    """
    curr_iter = trainer.logger.global_step
    image_outputs = torch.stack(image_outputs).squeeze()
    image_grid = vutils.make_grid(
        image_outputs, nrow=im_per_row, normalize=True, scale_each=True
    )
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()

    if comet_exp is not None:
        comet_exp.log_image(
            image_grid, name=f"{mode}_{domain}_{task}_{str(curr_iter)}", step=curr_iter,
        )


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

    # ----------------------------------
    # -----  Set Comet Experiment  -----
    # ----------------------------------
    exp = None
    if not args.no_comet:
        exp = Experiment(project_name="omnigan", auto_metric_logging=False)
        exp.log_parameters(flatten_opts(opts))
        # if args.note:
        #     exp.log_parameter("note", args.note)
        # with open(Path(opts.output_path) / "comet_url.txt", "w") as f:
        #     f.write(exp.url)
    # ------------------------
    # ----- Define model -----
    # ------------------------
    opts.data.loaders.batch_size = 1
    trainer = Trainer(opts)
    print("Constructed a trainer!")
    trainer.setup()
    print("Trainer setup compelete!")
    trainer.resume()
    print("Trainer resumed!")
    model = trainer.G
    model.eval()

    print("Model setting compelete!")

    # -------------------------------
    # -----  Transforms images  -----
    # -------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ----------------------------
    # -----  Iterate images  -----
    # ----------------------------

    # eval_folder(args.path_to_images, output_dir)

    # rootdir = args.path_to_images
    # writedir = args.output_dir

    # for root, subdirs, files in tqdm(os.walk(rootdir)):
    #     root = Path(root)
    #     subdirs = [Path(subdir) for subdir in subdirs]
    #     files = [Path(f) for f in files]
    #     has_imgs = False
    #     for f in files:
    #         if isimg(f):
    #             # read_path = root / f
    #             # rel_path = read_path.relative_to(rootdir)
    #             # write_path = writedir / rel_path
    #             # write_path.mkdir(parents=True, exist_ok=True)
    #             has_imgs = True
    #             break

    #     if has_imgs:
    #         print(f"Eval on {root}")
    #         rel_path = root.relative_to(rootdir)
    #         write_path = writedir / rel_path
    #         write_path.mkdir(parents=True, exist_ok=True)
    eval_folder(trainer, opts, args)
