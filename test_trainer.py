import atexit
from argparse import ArgumentParser

from comet_ml import Experiment
from comet_ml.api import API
import torch

import omnigan
from omnigan.utils import get_comet_rest_api_key
from pathlib import Path
from shutil import rmtree


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Colors:
    def _r(self, key, *args):
        return f"{key}{' '.join(args)}{bcolors.ENDC}"

    def ob(self, *args):
        return self._r(bcolors.OKBLUE, *args)

    def w(self, *args):
        return self._r(bcolors.WARNING, *args)

    def og(self, *args):
        return self._r(bcolors.OKGREEN, *args)

    def f(self, *args):
        return self._r(bcolors.FAIL, *args)

    def b(self, *args):
        return self._r(bcolors.BOLD, *args)

    def u(self, *args):
        return self._r(bcolors.UNDERLINE, *args)


def comet_handler(exp, api):
    def sub_handler():
        p = Colors()
        print()
        print(p.b(p.w("Deleting comet experiment")))
        api.delete_experiment(exp.get_key())

    return sub_handler


def print_start(desc):
    p = Colors()
    cdesc = p.b(p.ob(desc))
    title = "|  " + cdesc + "  |"
    line = "-" * (len(desc) + 6)
    print(f"{line}\n{title}\n{line}")


def print_end(desc):
    p = Colors()
    cdesc = p.b(p.og(desc))
    title = "|  " + cdesc + "  |"
    line = "-" * (len(desc) + 6)
    print(f"{line}\n{title}\n{line}")


def delete_on_exit(exp):
    rest_api_key = get_comet_rest_api_key()
    api = API(api_key=rest_api_key)
    atexit.register(comet_handler(exp, api))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--no_delete", action="store_true", default=False)
    parser.add_argument("--no_end_to_end", action="store_true", default=False)
    args = parser.parse_args()

    exp = Experiment(project_name="omnigan-test")
    if not args.no_delete:
        delete_on_exit(exp)

    prompt = Colors()

    opts = omnigan.utils.load_opts()
    opts.data.check_samples = False
    opts.train.fid.n_images = 5
    opts.comet.display_size = 5
    opts.tasks = ["m", "s", "d"]
    opts.domains = ["r", "s"]
    opts.data.loaders.num_workers = 4
    opts.data.loaders.batch_size = 2
    opts.data.max_samples = 9
    opts.train.epochs = 1
    opts.data.transforms[-1].new_size = 256

    # ---------------------------------
    # -----  MSD Trainer no Exp.  -----
    # ---------------------------------

    print_start("Running MSD no Exp")
    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=None,)
    trainer.functional_test_mode()
    trainer.setup()
    trainer.train()
    print_end("Done")

    del trainer
    torch.cuda.empty_cache()

    # ----------------------------------
    # -----  MSD Trainer with Exp  -----
    # ----------------------------------
    print_start("Running MSD with Exp")
    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=exp)
    trainer.functional_test_mode()
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.setup()
    trainer.train()
    print_end("Done")

    # -----------------------
    # -----  P trainer  -----
    # -----------------------
    print_start("Running P")
    opts.tasks = ["p"]
    opts.domains = ["rf"]
    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=exp)
    trainer.functional_test_mode()
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.setup()
    trainer.train()
    print_end("Done")

    # --------------------------------
    # -----  MSDP no end-to-end  -----
    # --------------------------------
    print_start("Running MSDP no end-to-end")
    opts.tasks = ["m", "s", "d", "p"]
    opts.domains = ["rf", "r", "s"]
    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=exp)
    trainer.functional_test_mode()
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.setup()
    trainer.train()
    print_end("Done")

    # ----------------------------------
    # -----  MSDP with end-to-end  -----
    # ----------------------------------
    print_start("Running MSDP with end-to-end")
    opts.tasks = ["m", "s", "d", "p"]
    opts.domains = ["rf", "r", "s"]
    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=exp)
    trainer.functional_test_mode()
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.use_pl4m = True
    trainer.setup()
    trainer.train()
    print_end("Done")
