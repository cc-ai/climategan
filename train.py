import logging
import os
from pathlib import Path
from time import sleep, time

import hydra
import yaml
from addict import Dict
from comet_ml import ExistingExperiment, Experiment
from omegaconf import OmegaConf

from climategan.trainer import Trainer
from climategan.utils import (
    comet_kwargs,
    copy_run_files,
    env_to_path,
    find_existing_training,
    flatten_opts,
    get_existing_comet_id,
    get_git_branch,
    get_git_revision_hash,
    get_increased_path,
    kill_job,
    load_opts,
    pprint,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

hydra_config_path = Path(__file__).resolve().parent / "shared/trainer/config.yaml"


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
    auto_resumed = {}

    config_path = args.config

    if hydra_opts.train.resume:
        out_ = str(env_to_path(hydra_opts.output_path))
        config_path = Path(out_) / "opts.yaml"
        if not config_path.exists():
            config_path = None
            print("WARNING: could not reuse the opts in {}".format(out_))

    default = args.default or Path(__file__).parent / "shared/trainer/defaults.yaml"

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(config_path, default=default, commandline_opts=hydra_opts)
    if args.resume:
        opts.train.resume = True

    opts.jobID = os.environ.get("SLURM_JOBID")
    opts.slurm_partition = os.environ.get("SLURM_JOB_PARTITION")
    opts.output_path = str(env_to_path(opts.output_path))
    print("Config output_path:", opts.output_path)

    exp = comet_previous_id = None

    # -------------------------------
    # -----  Check output_path  -----
    # -------------------------------

    # Auto-continue if same slurm job ID (=job was requeued)
    if not opts.train.resume and opts.train.auto_resume:
        print("\n\nTrying to auto-resume...")
        existing_path = find_existing_training(opts)
        if existing_path is not None and existing_path.exists():
            auto_resumed["original output_path"] = str(opts.output_path)
            auto_resumed["existing_path"] = str(existing_path)
            opts.train.resume = True
            opts.output_path = str(existing_path)

    # Still not resuming: creating new output path
    if not opts.train.resume:
        opts.output_path = str(get_increased_path(opts.output_path))
        Path(opts.output_path).mkdir(parents=True, exist_ok=True)

    # Copy the opts's sbatch_file to output_path
    copy_run_files(opts)
    # store git hash
    opts.git_hash = get_git_revision_hash()
    opts.git_branch = get_git_branch()

    if not args.no_comet:
        # ----------------------------------
        # -----  Set Comet Experiment  -----
        # ----------------------------------

        if opts.train.resume:
            # Is resuming: get existing comet exp id
            assert Path(opts.output_path).exists(), "Output_path does not exist"

            comet_previous_id = get_existing_comet_id(opts.output_path)
            # Continue existing experiment
            if comet_previous_id is None:
                print("WARNING could not retreive previous comet id")
                print(f"from {opts.output_path}")
            else:
                print("Continuing previous experiment", comet_previous_id)
                auto_resumed["continuing exp id"] = comet_previous_id
                exp = ExistingExperiment(
                    previous_experiment=comet_previous_id, **comet_kwargs
                )
                print("Comet Experiment resumed")

        if exp is None:
            # Create new experiment
            print("Starting new experiment")
            exp = Experiment(project_name="climategan", **comet_kwargs)
            exp.log_asset_folder(
                str(Path(__file__).parent / "climategan"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset(str(Path(__file__)))

        # Log note
        if args.note:
            exp.log_parameter("note", args.note)

        # Merge and log tags
        if args.comet_tags or opts.comet.tags:
            tags = set([f"branch:{opts.git_branch}"])
            if args.comet_tags:
                tags.update(args.comet_tags)
            if opts.comet.tags:
                tags.update(opts.comet.tags)
            opts.comet.tags = list(tags)
            print("Logging to comet.ml with tags", opts.comet.tags)
            exp.add_tags(opts.comet.tags)

        # Log all opts
        exp.log_parameters(flatten_opts(opts))
        if auto_resumed:
            exp.log_text("\n".join(f"{k:20}: {v}" for k, v in auto_resumed.items()))

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

    pprint("Running model in", opts.output_path)

    # -------------------
    # -----  Train  -----
    # -------------------

    trainer = Trainer(opts, comet_exp=exp, verbose=1)
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
