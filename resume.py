from pathlib import Path
from time import time, sleep
import os
from argparse import ArgumentParser

import yaml
from addict import Dict
from comet_ml import Experiment, ExistingExperiment

from omnigan.trainer import Trainer

from omnigan.utils import (
    env_to_path,
    flatten_opts,
    get_increased_path,
    comet_id_from_url,
    comet_kwargs,
    get_latest_path,
)


def get_latest_opts(path):
    """
    get latest opts dumped in path if they look like *opts*.yaml
    and were increased as
    opts.yaml < opts (1).yaml < opts (2).yaml etc.

    Args:
        path (str or pathlib.Path): where to look for opts

    Raises:
        ValueError: If no match for *opts*.yaml is found

    Returns:
        addict.Dict: loaded opts
    """
    path = Path(path)
    opts = get_latest_path(path / "opts.yaml")
    assert opts.exists()
    with opts.open("r") as f:
        return Dict(yaml.safe_load(f))


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


def main():
    """
    Opts prevalence:
        1. Load file specified in args.default (or shared/trainer/defaults.yaml
           if none is provided)
        2. Update with file specified in args.config (or no update if none is provided)
        3. Update with parsed command-line arguments

        e.g.
        `python train.py args.config=config/large-lr.yaml data.loaders.batch_size=10`
        loads defaults, overrides with values in large-lr.yaml and sets batch_size to 10
    """

    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--no_comet", action="store_true", default=False)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--comet_tags", nargs="+", default=[])

    args = parser.parse_args()
    resume_path = Path(env_to_path(args.path)).resolve()
    assert resume_path.exists()
    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = get_latest_opts(resume_path)
    opts.train.resume = True
    opts.output_path = str(env_to_path(opts.output_path))
    pprint("Continuing model in", opts.output_path)

    exp = comet_previous_id = comet_previous_path = None
    # -------------------------------
    # -----  Check output_path  -----
    # -------------------------------

    if not args.no_comet:
        # ----------------------------------
        # -----  Set Comet Experiment  -----
        # ----------------------------------
        comet_previous_path = Path(opts.output_path) / "comet_url.txt"
        if comet_previous_path.exists():
            with comet_previous_path.open("r") as f:
                url = f.read().strip()
                comet_previous_id = comet_id_from_url(url)

        # Continue existing experiment
        if comet_previous_id is None:
            print(
                "WARNING could not retreive previous comet id",
                f"from {comet_previous_path}",
            )
            exp = Experiment(project_name="omnigan", **comet_kwargs)
            exp.log_asset_folder(
                str(Path(__file__).parent / "omnigan"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset(str(Path(__file__)))
        else:
            exp = ExistingExperiment(
                previous_experiment=comet_previous_id, **comet_kwargs
            )

        # log job id
        opts.jobID = os.environ.get("SLURM_JOBID")

        # log note
        if args.note:
            exp.log_parameter("note", args.note)

        # merge and log tags from args and opts
        if args.comet_tags or opts.comet.tags:
            tags = set()
            if args.comet_tags:
                tags.update(args.comet_tags)
            if opts.comet.tags:
                tags.update(opts.comet.tags)
            opts.comet.tags = list(tags)
            exp.add_tags(opts.comet.tags)

        # log all opts
        exp.log_parameters(flatten_opts(opts))

        # allow some time for comet to get its url
        sleep(1)

        # Save comet exp url
        url_path = get_increased_path(Path(opts.output_path) / "comet_url.txt")
        with open(url_path, "w") as f:
            f.write(exp.url)

        # Save config file
        opts_path = get_increased_path(Path(opts.output_path) / "opts.yaml")
        with (opts_path).open("w") as f:
            yaml.safe_dump(opts.to_dict(), f)

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


if __name__ == "__main__":

    main()
