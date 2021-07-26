print("Imports...", end="", flush=True)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import atexit
import logging
from argparse import ArgumentParser
from copy import deepcopy

import comet_ml
import climategan
from comet_ml.api import API
from climategan.trainer import Trainer
from climategan.utils import get_comet_rest_api_key

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
import traceback

print("Done.")


def set_opts(opts, str_nested_key, value):
    """
    Changes an opts with nested keys:
    set_opts(addict.Dict(), "a.b.c", 2) == Dict({"a":{"b": {"c": 2}}})

    Args:
        opts (addict.Dict): opts whose values should be changed
        str_nested_key (str): nested keys joined on "."
        value (any): value to set to the nested keys of opts
    """
    keys = str_nested_key.split(".")
    o = opts
    for k in keys[:-1]:
        o = o[k]
    o[keys[-1]] = value


def set_conf(opts, conf):
    """
    Updates opts according to a test scenario's configuration dict.
    Ignores all keys starting with "__" which are used for the scenario
    but outside the opts

    Args:
        opts (addict.Dict): trainer options
        conf (dict): scenario's configuration
    """
    for k, v in conf.items():
        if k.startswith("__"):
            continue
        set_opts(opts, k, v)


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


def print_end(desc=None, ok=None):
    p = Colors()
    if ok and desc is None:
        desc = "Done"
        cdesc = p.b(p.og(desc))
    elif not ok and desc is None:
        desc = "! Fail !"
        cdesc = p.b(p.f(desc))
    elif desc is not None:
        cdesc = p.b(p.og(desc))
    else:
        desc = "Unknown"
        cdesc = desc

    title = "|  " + cdesc + "  |"
    line = "-" * (len(desc) + 6)
    print(f"{line}\n{title}\n{line}\n")


def delete_on_exit(exp):
    """
    Registers a callback to delete the comet exp at program exit

    Args:
        exp (comet_ml.Experiment): The exp to delete
    """
    rest_api_key = get_comet_rest_api_key()
    api = API(api_key=rest_api_key)
    atexit.register(comet_handler(exp, api))


if __name__ == "__main__":

    # -----------------------------
    # -----  Parse Arguments  -----
    # -----------------------------
    parser = ArgumentParser()
    parser.add_argument("--no_delete", action="store_true", default=False)
    parser.add_argument("--no_end_to_end", action="store_true", default=False)
    parser.add_argument("--include", "-i", nargs="+", default=[])
    parser.add_argument("--exclude", "-e", nargs="+", default=[])
    args = parser.parse_args()

    assert not (args.include and args.exclude), "Choose 1: include XOR exclude"

    include = set(int(i) for i in args.include)
    exclude = set(int(i) for i in args.exclude)
    if include:
        print("Including exclusively tests", " ".join(args.include))
    if exclude:
        print("Excluding tests", " ".join(args.exclude))

    # --------------------------------------
    # -----  Create global experiment  -----
    # --------------------------------------
    print("Creating comet Experiment...", end="", flush=True)
    global_exp = comet_ml.Experiment(
        project_name="climategan-test", display_summary_level=0
    )
    print("Done.")

    if not args.no_delete:
        delete_on_exit(global_exp)

    # prompt util for colors
    prompt = Colors()

    # -------------------------------------
    # -----  Base Test Scenario Opts  -----
    # -------------------------------------
    print("Loading opts...", end="", flush=True)
    base_opts = climategan.utils.load_opts()
    base_opts.data.check_samples = False
    base_opts.train.fid.n_images = 5
    base_opts.comet.display_size = 5
    base_opts.tasks = ["m", "s", "d"]
    base_opts.domains = ["r", "s"]
    base_opts.data.loaders.num_workers = 4
    base_opts.data.loaders.batch_size = 2
    base_opts.data.max_samples = 9
    base_opts.train.epochs = 1
    if isinstance(base_opts.data.transforms[-1].new_size, int):
        base_opts.data.transforms[-1].new_size = 256
    else:
        base_opts.data.transforms[-1].new_size.default = 256
    print("Done.")

    # --------------------------------------
    # -----  Configure Test Scenarios  -----
    # --------------------------------------

    # override any nested key in opts
    # create scenario-specific variables with __key
    # ALWAYS specify a __doc key to describe your scenario
    test_scenarios = [
        {"__use_comet": False, "__doc": "MSD no exp", "__verbose": 1},  # 0
        {"__doc": "MSD with exp"},  # 1
        {
            "__doc": "MSD no exp upsample_featuremaps",  # 2
            "__use_comet": False,
            "gen.d.upsample_featuremaps": True,
            "gen.s.upsample_featuremaps": True,
        },
        {"tasks": ["p"], "domains": ["rf"], "__doc": "Painter"},  # 3
        {
            "__doc": "M no exp low level feats",  # 4
            "__use_comet": False,
            "gen.m.use_low_level_feats": True,
            "gen.m.use_dada": False,
            "tasks": ["m"],
        },
        {
            "__doc": "MSD no exp deeplabv2",  # 5
            "__use_comet": False,
            "gen.encoder.architecture": "deeplabv2",
            "gen.s.architecture": "deeplabv2",
        },
        {
            "__doc": "MSDP no End-to-end",  # 6
            "domains": ["rf", "r", "s"],
            "tasks": ["m", "s", "d", "p"],
        },
        {
            "__doc": "MSDP inference only no exp",  # 7
            "__inference": True,
            "__use_comet": False,
            "domains": ["rf", "r", "s"],
            "tasks": ["m", "s", "d", "p"],
        },
        {
            "__doc": "MSDP with End-to-end",  # 8
            "__pl4m": True,
            "domains": ["rf", "r", "s"],
            "tasks": ["m", "s", "d", "p"],
        },
        {
            "__doc": "Kitti pretrain",  # 9
            "train.epochs": 2,
            "train.kitti.pretrain": True,
            "train.kitti.epochs": 1,
            "domains": ["kitti", "r", "s"],
            "train.kitti.batch_size": 2,
        },
        {"__doc": "Depth Dada archi", "gen.d.architecture": "dada"},  # 10
        {
            "__doc": "Depth Base archi",
            "gen.d.architecture": "base",
            "gen.m.use_dada": False,
            "gen.s.use_dada": False,
        },  # 11
        {
            "__doc": "Depth Base Classification",  # 12
            "gen.d.architecture": "base",
            "gen.d.classify.enable": True,
            "gen.m.use_dada": False,
            "gen.s.use_dada": False,
        },
        {
            "__doc": "MSD Resnet V3+ backbone",
            "gen.deeplabv3.backbone": "resnet",
        },  # 13
        {
            "__use_comet": False,
            "__doc": "MSD SPADE 12 (without x)",
            "__verbose": 1,
            "gen.m.use_spade": True,
            "gen.m.spade.cond_nc": 12,
        },  # 14
        {
            "__use_comet": False,
            "__doc": "MSD SPADE 15 (with x)",
            "__verbose": 1,
            "gen.m.use_spade": True,
            "gen.m.spade.cond_nc": 15,
        },  # 15
        {
            "__use_comet": False,
            "__doc": "Painter With Diff Augment",
            "__verbose": 1,
            "domains": ["rf"],
            "tasks": ["p"],
            "gen.p.diff_aug.use": True,
        },  # 15
        {
            "__use_comet": False,
            "__doc": "MSD DADA_s",
            "__verbose": 1,
            "gen.s.use_dada": True,
            "gen.m.use_dada": False,
        },  # 16
        {
            "__use_comet": False,
            "__doc": "MSD DADA_ms",
            "__verbose": 1,
            "gen.s.use_dada": True,
            "gen.m.use_dada": True,
        },  # 17
    ]

    n_confs = len(test_scenarios)

    fails = []
    successes = []

    # --------------------------------
    # -----  Run Test Scenarios  -----
    # --------------------------------

    for test_idx, conf in enumerate(test_scenarios):
        if test_idx in exclude or (include and test_idx not in include):
            reason = (
                "because it is in exclude"
                if test_idx in exclude
                else "because it is not in include"
            )
            print("Ignoring test", test_idx, reason)
            continue

        # copy base scenario opts
        test_opts = deepcopy(base_opts)
        # update with scenario configuration
        set_conf(test_opts, conf)

        # print scenario description
        print_start(
            f"[{test_idx}/{n_confs - 1}] "
            + conf.get("__doc", "WARNING: no __doc for test scenario")
        )
        print()

        comet = conf.get("__use_comet", True)
        pl4m = conf.get("__pl4m", False)
        inference = conf.get("__inference", False)
        verbose = conf.get("__verbose", 0)

        # set (or not) experiment
        test_exp = None
        if comet:
            test_exp = global_exp

        try:
            # create trainer
            trainer = Trainer(
                opts=test_opts,
                verbose=verbose,
                comet_exp=test_exp,
            )
            trainer.functional_test_mode()

            # set (or not) painter loss for masker (= end-to-end)
            if pl4m:
                trainer.use_pl4m = True

            # test training procedure
            trainer.setup(inference=inference)
            if not inference:
                trainer.train()

            successes.append(test_idx)
            ok = True
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            fails.append(test_idx)
            ok = False
        finally:
            print_end(ok=ok)

    print_end(desc="     -----   Summary   -----     ")
    if len(fails) == 0:
        print("•• All scenarios were successful")
    else:
        print(f"•• {len(successes)}/{len(test_scenarios)} successful tests")
        print(f"•• Failed test indices: {', '.join(map(str, fails))}")
