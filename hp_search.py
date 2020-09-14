import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\33[3m"
    BEIGE = "\33[36m"


def cols():
    try:
        col = os.get_terminal_size().columns
    except Exception:
        col = 50
    return col


def print_box(txt):
    txt = "|" + " " * 5 + txt + " " * 5 + "|"
    line = "-" * len(txt)
    empty = "|" + " " * (len(txt) - 2) + "|"
    print(line)
    print(empty)
    print(txt)
    print(empty)
    print(line)


def print_header(idx):
    b = bcolors.BOLD
    bl = bcolors.OKBLUE
    e = bcolors.ENDC
    char = "≡"
    c = cols()

    txt = " " * 20
    txt += f"{b}{bl}Run {idx}{e}"
    txt += " " * 20
    ln = len(txt) - len(b) - len(bl) - len(e)
    t = int(np.floor((c - ln) / 2))
    tt = int(np.ceil((c - ln) / 2))

    print(char * c)
    print(char * t + " " * ln + char * tt)
    print(char * t + txt + char * tt)
    print(char * t + " " * ln + char * tt)
    print(char * c)


def print_footer():
    c = cols()
    char = "﹎"
    print(char * (c // len(char)))
    print()
    print(" " * (c // 2) + "•" + " " * (c - c // 2 - 1))
    print()


def clean_arg(v):
    """
    chain cleaning function

    Args:
        v (any): arg to pass to train.py

    Returns:
        str: parsed value to string
    """
    return stringify_list(crop_float(quote_string(v)))


def stringify_list(v):
    """
    Stringify list (with double quotes) so that it can be passed a an argument
    to train.py's hydra command-line parsing

    Args:
        v (any): value to clean

    Returns:
        any: type of v, str if v was a list
    """
    if isinstance(v, list):
        return '"{}"'.format(str(v).replace('"', "'"))
    return v


def quote_string(v):
    """
    Add double quotes around string if it contains a " " or an =

    Args:
        v (any): value to clean

    Returns:
        any: type of v, quoted if v is a string with " " or =
    """
    if isinstance(v, str):
        if " " in v or "=" in v:
            return f'"{v}"'
    return v


def crop_float(v):
    """
    If v is a float, crop precision to 5 digits and return v as a str

    Args:
        v (any): value to crop if float

    Returns:
        any: cropped float as str if v is a float, original v otherwise
    """
    if isinstance(v, float):
        return f"{v:.5f}"
    return v


def compute_n_search(conf):
    """
    Compute the number of searchs to do if using -1 as n_search and using
    cartesian or sequential search

    Args:
        conf (dict): experimental configuration

    Returns:
        int: size of the cartesian product or length of longest sequential field
    """
    samples = defaultdict(list)
    for k, v in conf.items():
        if not isinstance(v, dict) or "sample" not in v:
            continue
        samples[v["sample"]].append(v)

    totals = []

    if "cartesian" in samples:
        total = 1
        for s in samples["cartesian"]:
            total *= len(s["from"])
        totals.append(total)
    if "sequential" in samples:
        total = max(map(len, [s["from"] for s in samples["sequential"]]))
        totals.append(total)

    if totals:
        return max(totals)

    raise ValueError(
        "Used n_search=-1 without any field being 'cartesian' or 'sequential'"
    )


def sample_param(sample_dict):
    """sample a value (hyperparameter) from the instruction in the
    sample dict:
    {
        "sample": "range | list",
        "from": [min, max, step] | [v0, v1, v2 etc.]
    }
    if range, as np.arange is used, "from" MUST be a list, but may contain
    only 1 (=min) or 2 (min and max) values, not necessarily 3

    Args:
        sample_dict (dict): instructions to sample a value

    Returns:
        scalar: sampled value
    """
    if not isinstance(sample_dict, dict) or "sample" not in sample_dict:
        return sample_dict

    if sample_dict["sample"] == "cartesian":
        assert isinstance(
            sample_dict["from"], list
        ), "{}'s `from` field MUST be a list, found {}".format(
            sample_dict["sample"], sample_dict["from"]
        )
        return "__cartesian__"

    if sample_dict["sample"] == "sequential":
        assert isinstance(
            sample_dict["from"], list
        ), "{}'s `from` field MUST be a list, found {}".format(
            sample_dict["sample"], sample_dict["from"]
        )
        return "__sequential__"

    if sample_dict["sample"] == "range":
        return np.random.choice(np.arange(*sample_dict["from"]))

    if sample_dict["sample"] == "list":
        return np.random.choice(sample_dict["from"])

    if sample_dict["sample"] == "uniform":
        return np.random.uniform(*sample_dict["from"])

    raise ValueError("Unknown sample type in dict " + str(sample_dict))


def sample_sequentials(sequential_keys, exp, idx):
    """
    Samples sequentially from the "from" values specified in each key of the
    experimental configuration which have sample == "sequential"
    Unlike `cartesian` sampling, `sequential` sampling iterates *independently*
    over each keys

    Args:
        sequential_keys (list): keys to be sampled sequentially
        exp (dict): experimental config
        idx (int): index of the current sample

    Returns:
        conf: sampled dict
    """
    conf = {}
    for k in sequential_keys:
        v = exp[k]["from"]
        conf[k] = v[idx % len(v)]
    return conf


def sample_cartesians(cartesian_keys, exp, idx):
    """
    Returns the `idx`th item in the cartesian product of all cartesian keys to
    be sampled.

    Args:
        cartesian_keys (list): keys in the experimental configuration that are to
        be used in the full cartesian product
        exp (dict): experimental configuration
        idx (int): index of the current sample

    Returns:
        dict: sampled point in the cartesian space (with keys = cartesian_keys)
    """
    conf = {}
    cartesian_values = [exp[key]["from"] for key in cartesian_keys]
    product = list(itertools.product(*cartesian_values))
    for k, v in zip(cartesian_keys, product[idx % len(product)]):
        conf[k] = v
    return conf


def resolve(hp_conf, nb):
    """
    Samples parameters parametrized in `exp`: should be a dict with
    values which fit `sample_params(dic)`'s API

    Args:
        exp (dict): experiment's parametrization
        nb  (int): number of experiments to sample

    Returns:
        dict: sampled configuration
    """
    if nb == -1:
        nb = compute_n_search(hp_conf)

    confs = []
    for idx in range(nb):
        conf = {}
        cartesians = []
        sequentials = []
        for k, v in hp_conf.items():
            candidate = sample_param(v)
            if candidate == "__cartesian__":
                cartesians.append(k)
            elif candidate == "__sequential__":
                sequentials.append(k)
            else:
                conf[k] = candidate
        if sequentials:
            conf.update(sample_sequentials(sequentials, hp_conf, idx))
        if cartesians:
            conf.update(sample_cartesians(cartesians, hp_conf, idx))
        confs.append(conf)
    return confs


def get_template_params(template):
    """
    extract args in template str as {arg}

    Args:
        template (str): sbatch template string

    Returns:
        list(str): Args required to format the template string
    """
    return map(
        lambda s: s.replace("{", "").replace("}", ""), re.findall("\{.*?\}", template)
    )


def read_hp(name):
    """
    Read hp search configuration from shared/experiment/
    specified with or without the .yaml extension

    Args:
        name (str): name of the template to find in shared/experiment/

    Returns:
        dict: file's loaded
    """
    if ".yaml" not in name:
        name += ".yaml"
    paths = []
    dirs = ["shared", "config"]
    for d in dirs:
        path = Path(__file__).parent / d / "experiment" / name
        if path.exists():
            paths.append(path)

    if len(paths) == 0:
        failed = [Path(__file__).parent / d / "experiment" for d in dirs]
        s = "Could not find search config {} in :\n".format(name)
        for fd in failed:
            s += str(fd) + "\nAvailable:\n"
            for ym in fd.glob("*.yaml"):
                s += "    " + ym.name + "\n"
        raise ValueError(s)

    if len(paths) == 2:
        print(
            "Warning: found 2 relevant files for search config:\n{}".format(
                "\n".join(paths)
            )
        )
        print("Using {}".format(paths[-1]))

    with paths[-1].open("r") as f:
        return yaml.safe_load(f)


def read_template(name):
    """
    Read template from shared/template/ specified with or without the .sh extension

    Args:
        name (str): name of the template to find in shared/template/

    Returns:
        str: file's content as 1 string
    """
    if ".sh" not in name:
        name += ".sh"
    paths = []
    dirs = ["shared", "config"]
    for d in dirs:
        path = Path(__file__).parent / d / "template" / name
        if path.exists():
            paths.append(path)

    if len(paths) == 0:
        failed = [Path(__file__).parent / d / "template" for d in dirs]
        s = "Could not find template {} in :\n".format(name)
        for fd in failed:
            s += str(fd) + "\nAvailable:\n"
            for ym in fd.glob("*.sh"):
                s += "    " + ym.name + "\n"
        raise ValueError(s)

    if len(paths) == 2:
        print("Warning: found 2 relevant template files:\n{}".format("\n".join(paths)))
        print("Using {}".format(paths[-1]))

    with paths[-1].open("r") as f:
        return f.read()


def is_sampled(key, conf):
    """
    Is a key sampled or constant? Returns true if conf is empty

    Args:
        key (str): key to check
        conf (dict): hyper parameter search configuration dict

    Returns:
        bool: key is sampled?
    """
    return not conf or (
        key in conf and isinstance(conf[key], dict) and "sample" in conf[key]
    )


if __name__ == "__main__":

    """
    Notes:
        * Must provide template name as template=name
        * `name`.sh should be in shared/template/
    """

    # -------------------------------
    # -----  Default Variables  -----
    # -------------------------------

    args = sys.argv[1:]
    command_output = ""
    user = os.environ.get("USER")
    home = os.environ.get("HOME")
    search_conf = {}
    dev = False
    escape = False
    verbose = False
    template_name = None
    hp_search_name = None
    hp_search_nb = None
    resume = None

    hp_search_private = set(["n_search", "template", "search"])

    sbatch_path = Path(home) / "omni_sbatch_latest.sh"

    # --------------------------
    # -----  Sanity Check  -----
    # --------------------------

    for arg in args:
        if "=" not in arg or " = " in arg:
            raise ValueError(
                "Args should be passed as `key=value`. Received `{}`".format(arg)
            )

    # --------------------------------
    # -----  Parse Command Line  -----
    # --------------------------------

    args_dict = {arg.split("=")[0]: arg.split("=")[1] for arg in args}

    assert "template" in args_dict, "Please specify template=xxx"
    template = read_template(args_dict["template"])
    template_dict = {k: None for k in get_template_params(template)}

    train_args = []
    for k, v in args_dict.items():

        if k == "verbose":
            if v != "0":
                verbose = True

        elif k == "sbatch_path":
            sbatch_path = v

        elif k == "dev":
            if v.lower() != "false":
                dev = True

        elif k == "escape":
            if v.lower() != "false":
                escape = True

        elif k == "template":
            template_name = v

        elif k == "search":
            hp_search_name = v

        elif k == "n_search":
            hp_search_nb = int(v)

        elif k == "resume":
            resume = f'"{v}"'
            template_dict[k] = f'"{v}"'

        elif k in template_dict:
            template_dict[k] = v

        else:
            train_args.append(f"{k}={v}")

    # -----------------------------------------
    # -----  Load Hyper-Parameter Search  -----
    # -----------------------------------------

    if hp_search_name is not None:
        search_conf = read_hp(hp_search_name)
        if "n_search" in search_conf and hp_search_nb is None:
            hp_search_nb = search_conf["n_search"]

        assert (
            hp_search_nb is not None
        ), "n_search should be specified in a yaml file or from the command line"

        hps = resolve(search_conf, hp_search_nb)

    else:
        hps = [None]

    # ---------------------------------
    # -----  Run All Experiments  -----
    # ---------------------------------
    for hp_idx, hp in enumerate(hps):

        # copy shared values
        tmp_template_dict = template_dict.copy()
        tmp_train_args = train_args.copy()
        tmp_train_args_dict = {
            arg.split("=")[0]: arg.split("=")[1] for arg in tmp_train_args
        }
        # override shared values with run-specific values for run hp_idx/n_search
        if hp is not None:
            for k, v in hp.items():
                if k == "resume" and resume is None:
                    resume = f'"{v}"'
                # hp-search params to ignore
                if k in hp_search_private:
                    continue

                if k == "codeloc":
                    v = re.escape(v)
                # override template params depending on exp config
                if k in tmp_template_dict:
                    if template_dict[k] is None or is_sampled(k, search_conf):
                        tmp_template_dict[k] = v
                # store sampled / specified params in current tmp_train_args_dict
                else:
                    if k in tmp_train_args_dict:
                        if is_sampled(k, search_conf):
                            # warn if key was specified from the command line
                            print(
                                "Warning",
                                "overriding commandline arg {} with hp value {}".format(
                                    k, v
                                ),
                            )
                            tmp_train_args_dict[k] = v
                    else:
                        tmp_train_args_dict[k] = v

        # create sbatch file where required
        sbatch_path = Path(sbatch_path).resolve()
        # format train.py's args and crop floats' precision to 5 digits
        tmp_template_dict["train_args"] = " ".join(
            sorted(
                [
                    "{}={}".format(k, clean_arg(v))
                    for k, v in tmp_train_args_dict.items()
                ]
            )
        )

        if "resume.py" in template and resume is None:
            raise ValueError("No `resume` value but using a resume.py template")

        # format template with clean dict (replace None with "")
        sbatch = template.format(
            **{
                k: v if v is not None else ""
                for k, v in tmp_template_dict.items()
                if k in template_dict
            }
        )

        # --------------------------------------
        # -----  Execute `sbatch` Command  -----
        # --------------------------------------
        print_header(hp_idx)
        if not dev:
            if sbatch_path.exists():
                print(f"Warning: overwriting {sbatch_path}")

            # write sbatch file
            with open(sbatch_path, "w") as f:
                f.write(sbatch)

            # escape special characters such as " " from sbatch_path's parent dir
            parent = str(sbatch_path.parent)
            if escape:
                parent = re.escape(parent)

            # create command to execute in a subprocess
            command = "sbatch {}".format(sbatch_path.name)
            # execute sbatch command & store output
            command_output = subprocess.run(
                command.split(), stdout=subprocess.PIPE, cwd=parent
            )
            command_output = "\n" + command_output.stdout.decode("utf-8") + "\n"

            print(f"Running from {parent}:")
            print(f"$ {command}")

        # ---------------------------------
        # -----  Summarize Execution  -----
        # ---------------------------------
        if verbose:
            print(bcolors.BEIGE + bcolors.ITALIC, "\n" + sbatch + bcolors.ENDC)
        if not dev:
            print_box(command_output.strip())

        print(
            "{}{}Summary{} {}:".format(
                bcolors.UNDERLINE,
                bcolors.OKGREEN,
                bcolors.ENDC,
                f"{bcolors.WARNING}(DEV){bcolors.ENDC}" if dev else "",
            )
        )
        print(
            "    "
            + "\n    ".join(
                "{:10}: {}".format(k, v) for k, v in tmp_template_dict.items()
            )
        )
        print_footer()
