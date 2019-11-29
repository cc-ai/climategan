import os
import re
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from copy import copy
import yaml
from addict import Dict


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="What configuration file to use to overwrite shared/defaults.yml",
    )
    parser.add_argument(
        "--comet", action="store_true", help="Use comet.ml to log experiment"
    )
    return parser.parse_args()


def load_opts(path):
    """Loads a configuration Dict from a yaml file

    for all decoder in gen.decoders, opts.gen.{decoder} is created from
    opts.gen.default and updated with existing specifications in opts.gen.{decoder}

    For instance if the only thing in decoder A which changes from default is the
    init_gain then you only need to set
    gen:
      default: ...
      A:
        init_gain: 0.1

    Args:
        path ([type]): [description]

    Returns:
        addict.Dict: the configuration object
    """
    path = Path(path).resolve()
    print("Loading opts from", str(path))
    with open(path, "r") as stream:
        try:
            opts = Dict(yaml.safe_load(stream))
            for mode in ["train", "val"]:
                for domain in opts.data.files[mode]:
                    opts.data.files[mode][domain] = str(
                        Path(opts.data.files.base) / opts.data.files[mode][domain]
                    )

            for k in opts.tasks:
                tmp = copy(opts.gen.default)
                if k in opts.gen:
                    tmp.update(opts.gen[k])
                opts.gen[k] = tmp

            for k in {"A", "T"} & set(opts.tasks):
                tmp = copy(opts.dis.default)
                if k in opts.dis:
                    tmp.update(opts.dis[k])
                opts.dis[k] = tmp

            return opts
        except yaml.YAMLError as exc:
            print(exc)


def get_git_revision_hash():
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def get_increased_path(path):
    """Retuns an increased path: if dir exists, returns `dir (1)`.
    If `dir (i)` exists, returns `dir (max(i) + 1)`

    get_increased_path("test").mkdir() creates `test/`
    then
    get_increased_path("test").mkdir() creates `test (1)/`
    etc.
    if `test (3)/` exists but not `test (2)/`, `test (4)/` is created so that indexes
    always increase

    Args:
        path (str or pathlib.Path): the file/directory which may already exist and would
            need to be increased

    Returns:
        pathlib.Path: increased path
    """
    fp = Path(path).resolve()
    f = str(fp)

    vals = []
    for n in fp.parent.glob("{}*".format(fp.name)):
        ms = list(re.finditer(r"^{} \(\d+\)$".format(f), str(n)))
        if ms:
            m = list(re.finditer(r"\(\d+\)$", str(n)))[0].group()
            vals.append(int(m.replace("(", "").replace(")", "")))
    if vals:
        ext = " ({})".format(max(vals) + 1)
    elif fp.exists():
        ext = " (1)"
    else:
        ext = ""

    return fp.parent / (fp.name + ext + fp.suffix)


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


def flatten_opts(opts):
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, Dict):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if isinstance(v[0], Dict):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def transforms_string(ts):
    return " -> ".join([t.__class__.__name__ for t in ts.transforms])
