from pathlib import Path
from time import time, sleep
import os

import hydra
import yaml
from addict import Dict
from comet_ml import Experiment, ExistingExperiment
from omegaconf import OmegaConf

from omnigan.trainer import Trainer

from omnigan.utils import (
    comet_id_from_url,
    comet_kwargs,
    env_to_path,
    flatten_opts,
    get_git_revision_hash,
    get_increased_path,
    load_opts,
    get_latest_path,
    copy_sbatch,
    merge,
    get_existing_jobID,
    kill_job,
)

hydra_config_path = Path(__file__).resolve().parent / "shared/trainer/config.yaml"


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


# requires hydra-core==0.11.3 and omegaconf==1.4.1
@hydra.main(config_path=hydra_config_path, strict=False)
def main(opts):
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

    hydra_opts = Dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)
    default = args.default or Path(__file__).parent / "shared/trainer/defaults.yaml"

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(args.config, default=default)
    opts = merge(hydra_opts, opts)
    if args.resume:
        opts.train.resume = True
    opts.jobID = os.environ.get("SLURM_JOBID")
    opts.output_path = str(env_to_path(opts.output_path))
    print("Config output_path:", opts.output_path)

    exp = comet_previous_id = comet_previous_path = None
    if not args.dev:
        # -------------------------------
        # -----  Check output_path  -----
        # -------------------------------

        # Auto-continue if same slurm job ID (=job was requeued)
        if not opts.train.resume:
            existing_jobID = get_existing_jobID(opts.output_path)
            print("Current jobID:", opts.jobID)
            try:
                if opts.jobID is not None and existing_jobID == opts.jobID:
                    opts.train.resume = True
                    print("Resuming based on jobID")
            except Exception as e:
                print("Could not resume", e)
                print("Continuing with opts.train.resume =", opts.train.resume)

        # Still not resuming: creating new output path
        if not opts.train.resume:
            opts.output_path = str(get_increased_path(opts.output_path))
            Path(opts.output_path).mkdir(parents=True, exist_ok=True)

        pprint("Running model in", opts.output_path)
        copy_sbatch(opts)

        # Is resuming: get existing comet exp id
        if opts.train.resume:
            assert Path(
                opts.output_path
            ).exists(), "Cannot resume: output_path does not exist"
            # load previous comet experiment id
            comet_previous_path = get_latest_path(
                Path(opts.output_path) / "comet_url.txt"
            )
            if comet_previous_path.exists():
                with comet_previous_path.open("r") as f:
                    url = f.read().strip()
                    comet_previous_id = comet_id_from_url(url)

        # store git hash
        opts.git_hash = get_git_revision_hash()

        if not args.no_comet:
            # ----------------------------------
            # -----  Set Comet Experiment  -----
            # ----------------------------------

            if opts.train.resume:
                # Continue existing experiment
                if comet_previous_id is None:
                    print("WARNING could not retreive previous comet id")
                    print(f"from {comet_previous_path}")
                else:
                    print("Continuing previous experiment", comet_previous_id)
                    exp = ExistingExperiment(
                        previous_experiment=comet_previous_id, **comet_kwargs
                    )

            if exp is None:
                # Create new experiment
                print("Starting new experiment")
                exp = Experiment(project_name="omnigan", **comet_kwargs)
                exp.log_asset_folder(
                    str(Path(__file__).parent / "omnigan"),
                    recursive=True,
                    log_file_name=True,
                )
                exp.log_asset(str(Path(__file__)))

            # Log note
            if args.note:
                exp.log_parameter("note", args.note)

            # Merge and log tags
            if args.comet_tags or opts.comet.tags:
                tags = set()
                if args.comet_tags:
                    tags.update(args.comet_tags)
                if opts.comet.tags:
                    tags.update(opts.comet.tags)
                opts.comet.tags = list(tags)
                print("Logging to comet.ml with tags", opts.comet.tags)
                exp.add_tags(opts.comet.tags)

            # Log all opts
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

    else:
        # ----------------------
        # -----  Dev Mode  -----
        # ----------------------
        pprint("> /!\ Development mode ON")
        print("Cropping data to 32")
        opts.data.transforms += [
            Dict({"name": "crop", "ignore": False, "height": 64, "width": 64})
        ]

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
    kill_job(opts.jobID)


if __name__ == "__main__":

    main()
