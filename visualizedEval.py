from comet_ml import Experiment
import torch
from omnigan.utils import load_opts, flatten_opts
from torch.nn.functional import sigmoid
from pathlib import Path
from argparse import ArgumentParser
from omnigan.trainer import Trainer
from omnigan.data import get_all_loaders, pil_image_loader
from torchvision import transforms as trsfs
import torchvision.utils as vutils


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
        default="/latest_ckpt.pth",
        type=str,
        help="Path to experiment folder containing checkpoints/latest_ckpt.pth",
    )
    parser.add_argument(
        "--no_output", action="store_true", help="If you don't want to store images to output_dir"
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
        "--valr_json",
        default="/network/tmp1/ccai/data/omnigan/base/"
        + "val_r_full_with_labelbox.json",
        type=str,
        help="The json file where you want to evaluate for real domain.",
    )
    parser.add_argument(
        "--no_comet", action="store_true", help="DON'T use comet.ml to log experiment"
    )
    return parser.parse_args()


def eval(trainer, opts, args):
    save_images = {}
    update_task = 'm' # only support masker evaluation now

    for i, multi_batch_tuple in enumerate(trainer.val_loaders):
        multi_domain_batch = {
            batch["domain"][0]: trainer.batch_to_device(batch)
            for batch in multi_batch_tuple
        }
        save_images[update_task] = []
        j = 0
        for batch_domain, batch in multi_domain_batch.items():
            print("Evaluating {}th image".format(j))
            j += 1
            if batch_domain == args.image_domain:
                file_name = batch["paths"]["m"][0].split("/")[-1]
                x = batch["data"]["x"]
                m = batch["data"]["m"]
                z = model.encode(x)
                mask = sigmoid(model.decoders[update_task](z))
                mask = mask.repeat(1, 3, 1, 1)
                task_saves = []

                task_saves.append(x * (1.0 - mask))
                task_saves.append(x * (1.0 - (mask > 0.1) * 1))
                task_saves.append(x * (1.0 - (mask > 0.5) * 1))
                task_saves.append(x * (1.0 - m.repeat(1, 3, 1, 1)))
                task_saves.append(mask)
                save_images[update_task].append(x)
                for im in task_saves:
                    save_images[update_task].append(im)
                if not args.no_output:
                    print("Saving image at {}".format(output_dir / file_name))
                    vutils.save_image(mask, output_dir / file_name)
        write_images(
            trainer,
            image_outputs=save_images[update_task],
            file_name=file_name,
            mode="val",
            domain=args.image_domain,
            task=update_task,
            im_per_row=opts.comet.im_per_row.get(update_task, 6),
            comet_exp=exp,
        )


def write_images(
    trainer, image_outputs, file_name, mode, domain, task, im_per_row=3, comet_exp=None
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
            image_grid,
            name=file_name + f"{mode}_{domain}_{task}_{str(curr_iter)}",
            step=curr_iter,
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
    if Path(args.checkpoint).suffix == '':
        opts.output_path = str(Path(args.checkpoint).resolve())
    elif Path(args.checkpoint).suffix.lower() == '.pth':
        opt.load_paths.m = str(Path(args.checkpoint).resolve())
        opts.output_path = "Loading a specific pth file, not using this option."
    if args.image_domain == "r":
        opts.data.files.val.r = args.valr_json
        opts.data.files.train.r = opts.data.files.val.r
        opts.data.files.train.s = opts.data.files.val.r
        opts.data.files.val.s = opts.data.files.val.r
    else:
        opts.data.files.train.r = opts.data.files.val.s
        opts.data.files.train.s = opts.data.files.val.s
        opts.data.filesval.r = opts.data.files.val.s
    opts.data.loaders.batch_size = 1
    # opts.val.visualize = True
    # ----------------------------------
    # -----  Set Comet Experiment  -----
    # ----------------------------------
    exp = None
    if not args.no_comet:
        exp = Experiment(project_name="omnigan", auto_metric_logging=False)
        exp.log_parameters(flatten_opts(opts))
    # ------------------------
    # ----- Define model -----
    # ------------------------
    print(args)
    opts.data.loaders.batch_size = 1
    opts.val.visualize = True
    trainer = Trainer(opts)
    print("Constructed a trainer!")
    trainer.setup()
    print("Trainer setup compelete!")
    trainer.resume()
    print("Trainer resumed!")
    model = trainer.G
    model.eval()
    print("Model setting compelete!")
    eval(trainer, opts, args)
