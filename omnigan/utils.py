"""All non-tensor utils
"""
import os
import re
import subprocess
import json
from copy import deepcopy
from pathlib import Path

import yaml
from addict import Dict


def load_opts(path=None, default=None):
    # TODO add assert: if deeplabv2 then res_dim = 2048
    """Loads a configuration Dict from 2 files:
    1. default files with shared values across runs and users
    2. an overriding file with run- and user-specific values

    Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields

    Returns:
        addict.Dict: options dictionnary, with overwritten default values
    """
    assert default or path

    if default is None:
        default_opts = Dict()
    else:
        with open(default, "r") as f:
            default_opts = Dict(yaml.safe_load(f))

    if path is None:
        overriding_opts = Dict()
    else:
        with open(path, "r") as f:
            overriding_opts = Dict(yaml.safe_load(f))

    default_opts.update(overriding_opts)

    default_opts.domains = []
    if "m" in default_opts.tasks:
        default_opts.domains.extend(["r", "s"])
    if "p" in default_opts.tasks:
        default_opts.domains.append("rf")
    if "simclr" in default_opts.tasks:
        default_opts.domains.append("r")
        if default_opts.train.latent_domain_adaptation:
            default_opts.domains.append("s")
    default_opts.domains = list(set(default_opts.domains))

    return set_data_paths(default_opts)


def set_data_paths(opts):
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base

    Args:
        opts (addict.Dict): options

    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val"]:
        for domain in opts.data.files[mode]:
            opts.data.files[mode][domain] = str(
                Path(opts.data.files.base) / opts.data.files[mode][domain]
            )

    return opts


def load_test_opts(test_file_path="config/trainer/local_tests.yaml"):
    """Returns the special opts set up for local tests
    Args:
        test_file_path (str, optional): Name of the file located in config/
            Defaults to "local_tests.yaml".

    Returns:
        addict.Dict: Opts loaded from defaults.yaml and updated from test_file_path
    """
    return load_opts(
        Path(__file__).parent.parent / f"{test_file_path}",
        default=Path(__file__).parent.parent / "shared/trainer/defaults.yaml",
    )


def get_git_revision_hash():
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def write_hash(path):
    hash_code = get_git_revision_hash()
    with open(path, "w") as f:
        f.write(hash_code)


def get_increased_path(path):
    """Returns an increased path: if dir exists, returns `dir (1)`.
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
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.

    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }

    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten

    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, (Dict, dict)):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if isinstance(v[0], (Dict, dict)):
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


def get_comet_rest_api_key(path_to_config_file=None):
    """Gets a comet.ml rest_api_key in the following order:
    * config file specified as argument
    * environment variable
    * .comet.config file in the current working diretory
    * .comet.config file in your home

    config files must have a line like `rest_api_key=<some api key>`

    Args:
        path_to_config_file (str or pathlib.Path, optional): config_file to use.
            Defaults to None.

    Raises:
        ValueError: can't find a file
        ValueError: can't find the key in a file

    Returns:
        str: your comet rest_api_key
    """
    if "COMET_REST_API_KEY" in os.environ and path_to_config_file is None:
        return os.environ["COMET_REST_API_KEY"]
    if path_to_config_file is not None:
        p = Path(path_to_config_file)
    else:
        p = Path() / ".comet.config"
        if not p.exists():
            p = Path.home() / ".comet.config"
            if not p.exists():
                raise ValueError("Unable to find your COMET_REST_API_KEY")
    with p.open("r") as f:
        for l in f:
            if "rest_api_key" in l:
                return l.strip().split("=")[-1].strip()
    raise ValueError("Unable to find your COMET_REST_API_KEY in {}".format(str(p)))


def get_files(dirName):
    # create a list of file and sub directories
    files = os.listdir(dirName)
    all_files = list()
    for entry in files:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            all_files = all_files + get_files(fullPath)
        else:
            all_files.append(fullPath)

    return all_files


def make_json_file(
    keys,
    addresses,
    splitter="/",  # for windows user, use "\\" instead of using "/"
    name_of_the_json_file="jsonfile.json",
):
    """
        How to use it?
    e.g.
    main(['x','m','d'], [
    '/network/tmp1/ccai/data/munit_dataset/trainA_size_1200/',
    '/network/tmp1/ccai/data/munit_dataset/seg_trainA_size_1200/',
    '/network/tmp1/ccai/data/munit_dataset/trainA_megadepth_resized/'
    ], 'train_r_resized.json')

    Args:
        keys (list): [description]
        addresses (list): [description]
        splitter (str, optional): [description]. Defaults to "/".
    """

    print("Please Make sure there is a file with the same name in each folder!")
    assert len(keys) == len(addresses), "keys and addresses must have the same length!"

    files = [get_files(addresses[j]) for j in range(len(keys))]

    file_address_map = {
        keys[j]: {
            ".".join(file.split(splitter)[-1].split(".")[:-1]): file
            for file in files[j]
        }
        for j in range(len(keys))
    }
    # The keys of the file_address_map are like 'x', 'm', 'd'...
    # The values of the file_address_map are a dictionary whose keys are the
    # filenames without extension whose values are the path of the filename
    # e.g. file_address_map =
    # {'x': {'A': 'path/to/trainA_size_1200/A.png', ...},
    #  'm': {'A': 'path/to/seg_trainA_size_1200/A.jpg',...}
    #  'd': {'A': 'path/to/trainA_megadepth_resized/A.bmp',...}
    # ...}

    dicts = []
    for file in files[0]:
        filename = file.split(splitter)[-1]  # the filename with 'x' extension
        filename_ = ".".join(filename.split(".")[:-1])  # the filename without extension
        tmp_dict = {}
        for i in range(len(keys)):
            tmp_dict[keys[i]] = file_address_map[keys[i]][filename_]
        dicts.append(tmp_dict)
    with open(name_of_the_json_file, "w", encoding="utf-8") as outfile:
        json.dump(dicts, outfile, ensure_ascii=False)


def sum_dict(dict1, dict2):
    """Add dict2 into dict1
    """
    for k, v in dict2.items():
        if not isinstance(v, dict):
            dict1[k] += v
        else:
            sum_dict(dict1[k], dict2[k])
    return dict1


def div_dict(dict1, div_by):
    """Divide elements of dict1 by div_by
    """
    for k, v in dict1.items():
        if not isinstance(v, dict):
            dict1[k] /= div_by
        else:
            div_dict(dict1[k], div_by)
    return dict1
