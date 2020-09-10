from pathlib import Path
from time import time, sleep

import hydra
import yaml
from addict import Dict
from comet_ml import Experiment
from omegaconf import OmegaConf

from omnigan.trainer import Trainer

from omnigan.utils import env_to_path, flatten_opts, get_increased_path, load_opts

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
    opts.update(hydra_opts)
    if args.resume:
        opts.train.resume = True
    opts.output_path = str(env_to_path(opts.output_path))

    if not opts.train.resume:
        opts.output_path = str(get_increased_path(opts.output_path))
    pprint("Running model in", opts.output_path)

    exp = None
    if not args.dev:
        # -------------------------------
        # -----  Check output_path  -----
        # -------------------------------
        if opts.train.resume:
            Path(opts.output_path).mkdir(exist_ok=True, parents=True)
        else:
            assert not Path(opts.output_path).exists()
            Path(opts.output_path).mkdir()

        # Save config file
        # TODO what if resuming? re-dump?
        # print("opts: ", opts.to_dict())
        with (Path(opts.output_path) / "opts.yaml").open("w") as f:
            yaml.safe_dump(
                opts.to_dict(), f,
            )

        if not args.no_comet:
            # ----------------------------------
            # -----  Set Comet Experiment  -----
            # ----------------------------------
            exp = Experiment(project_name="omnigan", auto_metric_logging=False)
            exp.log_parameters(flatten_opts(opts))
            if args.note:
                exp.log_parameter("note", args.note)
            if args.comet_tags:
                exp.add_tags(list(args.comet_tags))
            sleep(1)
            with open(Path(opts.output_path) / "comet_url.txt", "w") as f:
                f.write(exp.url)
    else:
        # ----------------------
        # -----  Dev Mode  -----
        # ----------------------
        pprint("> /!\ Development mode ON")
        print("Cropping data to 32")
        opts.data.transforms += [
            Dict({"name": "crop", "ignore": False, "height": 32, "width": 32})
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


if __name__ == "__main__":

    main()
