import datetime
import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def flatten_conf(conf, to={}, parents=[]):
    """
    Flattens a configuration dict: nested dictionaries are flattened
    as key1.key2.key3 = value

    conf.yaml:
    ```yaml
    a: 1
    b:
        c: 2
        d:
            e: 3
        g:
            sample: sequential
            from: [4, 5]
    ```

    Is flattened to

    {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3,
        "b.g": {
            "sample": "sequential",
            "from": [4, 5]
        }
    }

    Does not affect sampling dicts.

    Args:
        conf (dict): the configuration to flatten
        new (dict, optional): the target flatenned dict. Defaults to {}.
        parents (list, optional): a final value's list of parents. Defaults to [].
    """
    for k, v in conf.items():
        if isinstance(v, dict) and "sample" not in v:
            flatten_conf(v, to, parents + [k])
        else:
            new_k = ".".join([str(p) for p in parents + [k]])
            to[new_k] = v


def env_to_path(path):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable

    """
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


class C:
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


def escape_path(path):
    p = str(path)
    return p.replace(" ", "\ ").replace("(", "\(").replace(")", "\)")  # noqa: W605


def warn(*args, **kwargs):
    print("{}{}{}".format(C.WARNING, " ".join(args), C.ENDC), **kwargs)


def parse_jobID(command_output):
    """
    get job id from successful sbatch command output like
    `Submitted batch job 599583`

    Args:
        command_output (str): sbatch command's output

    Returns:
        int: the slurm job's ID
    """
    command_output = command_output.strip()
    if isinstance(command_output, str):
        if "Submitted batch job" in command_output:
            return int(command_output.split()[-1])

    return -1


def now():
    return str(datetime.datetime.now()).replace(" ", "_")


def cols():
    try:
        col = os.get_terminal_size().columns
    except Exception:
        col = 50
    return col


def print_box(txt):
    if not txt:
        txt = "{}{}ERROR ⇪{}".format(C.BOLD, C.FAIL, C.ENDC)
        lt = 7
    else:
        lt = len(txt)
    nlt = lt + 12
    txt = "|" + " " * 5 + txt + " " * 5 + "|"
    line = "-" * nlt
    empty = "|" + " " * (nlt - 2) + "|"
    print(line)
    print(empty)
    print(txt)
    print(empty)
    print(line)


def print_header(idx):
    b = C.BOLD
    bl = C.OKBLUE
    e = C.ENDC
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
    print()
    print(char * (c // len(char)))
    print()
    print(" " * (c // 2) + "•" + " " * (c - c // 2 - 1))
    print()


def extend_summary(summary, tmp_train_args_dict, tmp_template_dict, exclude=[]):
    exclude = set(exclude)
    if summary is None:
        summary = defaultdict(list)
    for k, v in tmp_template_dict.items():
        if k not in exclude:
            summary[k].append(v)
    for k, v in tmp_train_args_dict.items():
        if k not in exclude:
            if isinstance(v, list):
                v = str(v)
            summary[k].append(v)
    return summary


def search_summary_table(summary, summary_dir=None):
    # filter out constant values
    summary = {k: v for k, v in summary.items() if len(set(v)) > 1}

    # if everything is constant: no summary
    if not summary:
        return None, None

    # find number of searches
    n_searches = len(list(summary.values())[0])

    # print section title
    print(
        "{}{}{}Varying values across {} experiments:{}\n".format(
            C.OKBLUE,
            C.BOLD,
            C.UNDERLINE,
            n_searches,
            C.ENDC,
        )
    )

    # first column holds the Exp. number
    first_col = {
        "len": 8,  # length of a column, to split columns according to terminal width
        "str": ["| Exp. |", "|:----:|"]
        + [
            "| {0:^{1}} |".format(i, 4) for i in range(n_searches)
        ],  # list of values to print
    }

    print_columns = [[first_col]]
    file_columns = [first_col]
    for k in sorted(summary.keys()):
        v = summary[k]
        col_title = f" {k} |"
        col_blank_line = f":{'-' * len(k)}-|"
        col_values = [
            " {0:{1}} |".format(
                crop_string(
                    str(crop_float(v[idx], min([5, len(k) - 2]))), len(k)
                ),  # crop floats and long strings
                len(k),
            )
            for idx in range(len(v))
        ]

        # create column object
        col = {"len": len(k) + 3, "str": [col_title, col_blank_line] + col_values}

        # if adding a new column would overflow the terminal and mess up printing, start
        # new set of columns
        if sum(c["len"] for c in print_columns[-1]) + col["len"] >= cols():
            print_columns.append([first_col])

        # store current column to latest group of columns
        print_columns[-1].append(col)
        file_columns.append(col)

    print_table = ""
    # print each column group individually
    for colgroup in print_columns:
        # print columns line by line
        for i in range(n_searches + 2):
            # get value of column for current line i
            for col in colgroup:
                print_table += col["str"][i]
            # next line for current columns
            print_table += "\n"

        # new lines for new column group
        print_table += "\n"

    file_table = ""
    for i in range(n_searches + 2):
        # get value of column for current line i
        for col in file_columns:
            file_table += col["str"][i]
        # next line for current columns
        file_table += "\n"

    summary_path = None
    if summary_dir is not None:
        summary_path = summary_dir / (now() + ".md")
        with summary_path.open("w") as f:
            f.write(file_table.strip())

    return print_table, summary_path


def clean_arg(v):
    """
    chain cleaning function

    Args:
        v (any): arg to pass to train.py

    Returns:
        str: parsed value to string
    """
    return stringify_list(crop_float(quote_string(resolve_env(v))))


def resolve_env(v):
    """
    resolve env variables in paths

    Args:
        v (any): arg to pass to train.py

    Returns:
        str: try and resolve an env variable
    """
    if isinstance(v, str):
        try:
            if "$" in v:
                if "/" in v:
                    v = env_to_path(v)
                else:
                    _v = os.environ.get(v)
                    if _v is not None:
                        v = _v
        except Exception:
            pass
    return v


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
    if isinstance(v, str):
        if v.startswith("[") and v.endswith("]"):
            return f'"{v}"'
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


def crop_float(v, k=5):
    """
    If v is a float, crop precision to 5 digits and return v as a str

    Args:
        v (any): value to crop if float

    Returns:
        any: cropped float as str if v is a float, original v otherwise
    """
    if isinstance(v, float):
        return "{0:.{1}g}".format(v, k)
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


def crop_string(s, k=10):
    if len(s) <= k:
        return s
    else:
        return s[: k - 2] + ".."


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
        lambda s: s.replace("{", "").replace("}", ""),
        re.findall("\{.*?\}", template),  # noqa: W605
    )


def read_exp_conf(name):
    """
    Read hp search configuration from shared/experiment/
    specified with or without the .yaml extension

    Args:
        name (str): name of the template to find in shared/experiment/

    Returns:
        Tuple(Path, dict): file path and loaded dict
    """
    if ".yaml" not in name:
        name += ".yaml"
    paths = []
    dirs = ["shared", "config"]
    for d in dirs:
        path = Path(__file__).parent / d / "experiment" / name
        if path.exists():
            paths.append(path.resolve())

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
        conf = yaml.safe_load(f)

    flat_conf = {}
    flatten_conf(conf, to=flat_conf)

    return (paths[-1], flat_conf)


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
    exp_conf = {}
    dev = False
    escape = False
    verbose = False
    template_name = None
    hp_exp_name = None
    hp_search_nb = None
    exp_path = None
    resume = None
    force_sbatchs = False
    sbatch_base = Path(home) / "climategan_sbatchs"
    summary_dir = Path(home) / "climategan_exp_summaries"

    hp_search_private = set(["n_search", "template", "search", "summary_dir"])

    sbatch_path = "hash"

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

        elif k == "sbatch_base":
            sbatch_base = Path(v).resolve()

        elif k == "force_sbatchs":
            force_sbatchs = v.lower() == "true"

        elif k == "dev":
            if v.lower() != "false":
                dev = True

        elif k == "escape":
            if v.lower() != "false":
                escape = True

        elif k == "template":
            template_name = v

        elif k == "exp":
            hp_exp_name = v

        elif k == "n_search":
            hp_search_nb = int(v)

        elif k == "resume":
            resume = f'"{v}"'
            template_dict[k] = f'"{v}"'

        elif k == "summary_dir":
            if v.lower() == "none":
                summary_dir = None
            else:
                summary_dir = Path(v)

        elif k in template_dict:
            template_dict[k] = v

        else:
            train_args.append(f"{k}={v}")

    # ------------------------------------
    # -----  Load Experiment Config  -----
    # ------------------------------------

    if hp_exp_name is not None:
        exp_path, exp_conf = read_exp_conf(hp_exp_name)
        if "n_search" in exp_conf and hp_search_nb is None:
            hp_search_nb = exp_conf["n_search"]

        assert (
            hp_search_nb is not None
        ), "n_search should be specified in a yaml file or from the command line"

        hps = resolve(exp_conf, hp_search_nb)

    else:
        hps = [None]

    # ---------------------------------
    # -----  Run All Experiments  -----
    # ---------------------------------
    if summary_dir is not None:
        summary_dir.mkdir(exist_ok=True, parents=True)
    summary = None

    for hp_idx, hp in enumerate(hps):

        # copy shared values
        tmp_template_dict = template_dict.copy()
        tmp_train_args = train_args.copy()
        tmp_train_args_dict = {
            arg.split("=")[0]: arg.split("=")[1] for arg in tmp_train_args
        }
        print_header(hp_idx)
        # override shared values with run-specific values for run hp_idx/n_search
        if hp is not None:
            for k, v in hp.items():
                if k == "resume" and resume is None:
                    resume = f'"{v}"'
                # hp-search params to ignore
                if k in hp_search_private:
                    continue

                if k == "codeloc":
                    v = escape_path(v)

                if k == "output":
                    Path(v).parent.mkdir(parents=True, exist_ok=True)

                # override template params depending on exp config
                if k in tmp_template_dict:
                    if template_dict[k] is None or is_sampled(k, exp_conf):
                        tmp_template_dict[k] = v
                # store sampled / specified params in current tmp_train_args_dict
                else:
                    if k in tmp_train_args_dict:
                        if is_sampled(k, exp_conf):
                            # warn if key was specified from the command line
                            tv = tmp_train_args_dict[k]
                            warn(
                                "\nWarning: overriding sampled config-file arg",
                                "{} to command-line value {}\n".format(k, tv),
                            )
                    else:
                        tmp_train_args_dict[k] = v

        # create sbatch file where required
        tmp_sbatch_path = None
        if sbatch_path == "hash":
            tmp_sbatch_name = "" if hp_exp_name is None else hp_exp_name[:14] + "_"
            tmp_sbatch_name += now() + ".sh"
            tmp_sbatch_path = sbatch_base / tmp_sbatch_name
            tmp_sbatch_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_train_args_dict["sbatch_file"] = str(tmp_sbatch_path)
            tmp_train_args_dict["exp_file"] = str(exp_path)
        else:
            tmp_sbatch_path = Path(sbatch_path).resolve()

        summary = extend_summary(
            summary, tmp_train_args_dict, tmp_template_dict, exclude=["sbatch_file"]
        )

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
        if not dev or force_sbatchs:
            if tmp_sbatch_path.exists():
                print(f"Warning: overwriting {sbatch_path}")

            # write sbatch file
            with open(tmp_sbatch_path, "w") as f:
                f.write(sbatch)

        if not dev:
            # escape special characters such as " " from sbatch_path's parent dir
            parent = str(tmp_sbatch_path.parent)
            if escape:
                parent = escape_path(parent)

            # create command to execute in a subprocess
            command = "sbatch {}".format(tmp_sbatch_path.name)
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
            print(C.BEIGE + C.ITALIC, "\n" + sbatch + C.ENDC)
        if not dev:
            print_box(command_output.strip())
            jobID = parse_jobID(command_output.strip())
            summary["Slurm JOBID"].append(jobID)

        summary["Comet Link"].append(f"[{hp_idx}][{hp_idx}]")

        print(
            "{}{}Summary{} {}:".format(
                C.UNDERLINE,
                C.OKGREEN,
                C.ENDC,
                f"{C.WARNING}(DEV){C.ENDC}" if dev else "",
            )
        )
        print(
            "    "
            + "\n    ".join(
                "{:10}: {}".format(k, v) for k, v in tmp_template_dict.items()
            )
        )
        print_footer()

    print(f"\nRan a total of {len(hps)} jobs{' in dev mode.' if dev else '.'}\n")

    table, sum_path = search_summary_table(summary, summary_dir if not dev else None)
    if table is not None:
        print(table)
        print(
            "Add `[i]: https://...` at the end of a markdown document",
            "to fill in the comet links.\n",
        )
        if summary_dir is None:
            print("Add summary_dir=path to store the printed markdown table ⇪")
        else:
            print("Saved table in", str(sum_path))

    if not dev:
        print(
            "Cancel entire experiment? \n$ scancel",
            " ".join(map(str, summary["Slurm JOBID"])),
        )
