from pathlib import Path
from time import time

import hydra
import yaml
from addict import Dict
from comet_ml import Experiment
from omegaconf import OmegaConf

from omnigan.trainer import Trainer

from omnigan.utils import (
    env_to_path,
    flatten_opts,
    get_increased_path,
    load_opts,
    adventv2EntropySplit,
    switch_data,
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


@hydra.main(config_path=hydra_config_path)
def main(opts):
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    opts = Dict(OmegaConf.to_container(opts))
    args = opts.args

    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    opts = load_opts(args.config, default=opts)
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
            Path(opts.output_path).mkdir(exist_ok=True)
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

    # ------------------------------------------
    # -----  Check if auto adventv2 works  -----
    # ------------------------------------------
    is_auto_adventv2 = opts.train.lambdas.advent.is_auto_adventv2
    if is_auto_adventv2:
        assert opts.tasks == [
            "m"
        ], "Auto adventv2 only works if mask is the only task to be trained!"
        opts.train.epochs = opts.train.lambdas.advent.stage_one_epochs
        if opts.train.resume:
            if opts.train.lambdas.advent.stage_one_epochs == 0:
                print("Continue on stage two training")
            else:
                print("Continue on stage one training")

    # -------------------
    # -----  Train  -----
    # -------------------
    trainer = Trainer(opts, comet_exp=exp)
    trainer.logger.time.start_time = time()
    trainer.setup()
    # start training if the expected training epochs is not 0
    if opts.train.epochs != 0:
        trainer.train()
    if is_auto_adventv2:
        if not opts.train.lambdas.advent.split_prepared:
            adventv2EntropySplit(trainer, verbose=0)
        else:
            print("Skip generating entropy split.")
        print("Start stage two training!")
        trainer.logger.epoch += 1
        trainer.opts = switch_data(opts)
        trainer.train()

    # -----------------------------
    # -----  End of training  -----
    # -----------------------------

    pprint("Done training")


if __name__ == "__main__":

    main()
