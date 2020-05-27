from comet_ml import Experiment

from argparse import ArgumentParser
from pathlib import Path
from time import time

from addict import Dict

from omnigan.trainer import Trainer
from omnigan.utils import env_to_path, flatten_opts, get_increased_path, load_opts


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./config/local_tests.yaml",
        type=str,
        help="What configuration file to use to overwrite shared/defaults.yaml",
    )
    parser.add_argument(
        "--exp_desc", default="", type=str, help="Description of the experiment",
    )
    parser.add_argument(
        "--note", default="", type=str, help="Note about this training",
    )
    parser.add_argument(
        "--no_comet", action="store_true", help="DON'T use comet.ml to log experiment"
    )
    parser.add_argument(
        "--dev_mode",
        action="store_true",
        default=False,
        help="Run this script in development mode",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Repeatable flag to add tags to the comet exp (--tag a --tag b ...)",
    )

    return parser.parse_args()


def pprint(*args):
    txt = " ".join(map(str, args))
    col = "====="
    space = "   "
    head_size = 2
    header = "\n".join(["=" * (len(txt) + 2 * (len(col) + len(space)))] * head_size)
    empty = "{}{}{}{}{}".format(col, space, " " * (len(txt)), space, col)
    print()
    print(header)
    print(empty)
    print("{}{}{}{}{}".format(col, space, txt, space, col))
    print(empty)
    print(header)
    print()


if __name__ == "__main__":

    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(Path(args.config), default="./shared/trainer/defaults.yaml")
    opts.output_path = env_to_path(opts.output_path)
    opts.output_path = get_increased_path(opts.output_path)
    pprint("Running model in", opts.output_path)
    if args.dev_mode:
        assert not Path(opts.output_path).exists()
    else:
        Path(opts.output_path).mkdir()

    # ----------------------------------
    # -----  Set Comet Experiment  -----
    # ----------------------------------

    exp = None
    if not args.no_comet and not args.dev_mode:
        exp = Experiment(project_name="omnigan", auto_metric_logging=False)
        exp.log_parameters(flatten_opts(opts))
        if args.exp_desc:
            exp.log_parameter("exp_desc", args.exp_desc)
        if args.note:
            exp.log_parameter("note", args.note)
        with open(Path(opts.output_path) / "comet_url.txt", "w") as f:
            f.write(exp.url)

    # ----------------------
    # -----  Dev Mode  -----
    # ----------------------

    if args.dev_mode:
        pprint("> /!\ Development mode ON")
        print("Cropping data to 32")
        opts.data.transforms += [Dict({"name": "crop", "ignore": False, "height": 32, "width": 32})]

    # -------------------
    # -----  Train  -----
    # -------------------

    trainer = Trainer(opts, comet_exp=exp)
    trainer.logger.time.start_time = time()
    trainer.setup()
    trainer.train()

    # -----------------------------
    # -----  End of training  -----
    # -----------------------------

    pprint("Done training")
