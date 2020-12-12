from argparse import ArgumentParser
from pathlib import Path

from addict import Dict
from comet_ml import Experiment  # noqa: F401 -> keep even if unused

from omnigan.data import get_loader
from omnigan.trainer import Trainer
from omnigan.utils import flatten_opts


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--resume_path", required=True, type=str, help="Path to the trainer to resume",
    )
    parser.add_argument(
        "--image_domain",
        default="r",
        type=str,
        help="Domain of images in path_to_images, can be 'r' or 's'",
    )
    parser.add_argument(
        "--val_r_json",
        default="/network/tmp1/ccai/data/omnigan/base/"
        + "val_r_full_with_labelbox.json",
        type=str,
        help="The json file where you want to evaluate for real domain.",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        nargs="+",
        help="list of tasks to eval. eg: `-t m s`",
        default=["m"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()
    resume_path = Path(args.resume_path).expanduser().resolve()
    assert resume_path.exists()

    image_domain = args.image_domain
    assert image_domain in {"r", "s", "rf", "kitti"}

    overrides = Dict()
    overrides.data.loaders.batch_size = 1
    overrides.comet.rows_per_log = 1
    overrides.tasks = args.tasks
    if args.val_r_json:
        val_r_json_path = Path(args.val_r_json).expanduser().resolve()
        assert val_r_json_path.exists()
        overrides.data.files.val[image_domain] = str(val_r_json_path)

    trainer = Trainer.resume_from_path(
        resume_path, overrides=overrides, inference=True, new_exp=True
    )
    trainer.exp.log_parameters(flatten_opts(trainer.opts))
    trainer.all_loaders = {
        "val": {image_domain: get_loader("val", image_domain, trainer.opts)}
    }
    trainer.set_display_images(True)
    trainer.logger.log_comet_images("val", image_domain)
