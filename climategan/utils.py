"""All non-tensor utils
"""
import contextlib
import datetime
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from os.path import expandvars
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import uuid4

import numpy as np
import torch
import yaml
from addict import Dict
from comet_ml import Experiment

comet_kwargs = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute

    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def copy_run_files(opts: Dict) -> None:
    """
    Copy the opts's sbatch_file to output_path

    Args:
        opts (addict.Dict): options
    """
    if opts.sbatch_file:
        p = resolve(opts.sbatch_file)
        if p.exists():
            o = resolve(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)
    if opts.exp_file:
        p = resolve(opts.exp_file)
        if p.exists():
            o = resolve(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)


def merge(
    source: Union[dict, Dict], destination: Union[dict, Dict]
) -> Union[dict, Dict]:
    """
    run me with nosetests --with-doctest file.py
    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == {
        'first' : {
            'all_rows' : { '
                pass' : 'dog',
                'fail' : 'cat',
                'number' : '5'
            }
        }
    }
    True
    """
    for key, value in source.items():
        try:
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                merge(value, node)
            else:
                if isinstance(destination, dict):
                    destination[key] = value
                else:
                    destination = {key: value}
        except TypeError as e:
            print(traceback.format_exc())
            print(">>>", source)
            print(">>>", destination)
            print(">>>", key)
            print(">>>", value)
            raise Exception(e)

    return destination


def load_opts(
    path: Optional[Union[str, Path]] = None,
    default: Optional[Union[str, Path, dict, Dict]] = None,
    commandline_opts: Optional[Union[Dict, dict]] = None,
) -> Dict:
    """Loadsize a configuration Dict from 2 files:
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

    if path is None and default is None:
        path = (
            resolve(Path(__file__)).parent.parent
            / "shared"
            / "trainer"
            / "defaults.yaml"
        )

    if path:
        path = resolve(path)

    if default is None:
        default_opts = {}
    else:
        if isinstance(default, (str, Path)):
            with open(default, "r") as f:
                default_opts = yaml.safe_load(f)
        else:
            default_opts = dict(default)

    if path is None:
        overriding_opts = {}
    else:
        with open(path, "r") as f:
            overriding_opts = yaml.safe_load(f) or {}

    opts = Dict(merge(overriding_opts, default_opts))

    if commandline_opts is not None and isinstance(commandline_opts, dict):
        opts = Dict(merge(commandline_opts, opts))

    if opts.train.kitti.pretrained:
        assert "kitti" in opts.data.files.train
        assert "kitti" in opts.data.files.val
        assert opts.train.kitti.epochs > 0

    opts.domains = []
    if "m" in opts.tasks or "s" in opts.tasks or "d" in opts.tasks:
        opts.domains.extend(["r", "s"])
    if "p" in opts.tasks:
        opts.domains.append("rf")
    if opts.train.kitti.pretrain:
        opts.domains.append("kitti")

    opts.domains = list(set(opts.domains))

    if "s" in opts.tasks:
        if opts.gen.encoder.architecture != opts.gen.s.architecture:
            print(
                "WARNING: segmentation encoder and decoder architectures do not match"
            )
            print(
                "Encoder: {} <> Decoder: {}".format(
                    opts.gen.encoder.architecture, opts.gen.s.architecture
                )
            )
    if opts.gen.m.use_spade:
        if "d" not in opts.tasks or "s" not in opts.tasks:
            raise ValueError(
                "opts.gen.m.use_spade is True so tasks MUST include"
                + "both d and s, but received {}".format(opts.tasks)
            )
        if opts.gen.d.classify.enable:
            raise ValueError(
                "opts.gen.m.use_spade is True but using D as a classifier"
                + " which is a non-implemented combination"
            )

    if opts.gen.s.depth_feat_fusion is True or opts.gen.s.depth_dada_fusion is True:
        opts.gen.s.use_dada = True

    events_path = (
        resolve(Path(__file__)).parent.parent / "shared" / "trainer" / "events.yaml"
    )
    if events_path.exists():
        with events_path.open("r") as f:
            events_dict = yaml.safe_load(f)
        events_dict = Dict(events_dict)
        opts.events = events_dict

    return set_data_paths(opts)


def set_data_paths(opts: Dict) -> Dict:
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base

    Args:
        opts (addict.Dict): options

    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val"]:
        for domain in opts.data.files[mode]:
            if opts.data.files.base and not opts.data.files[mode][domain].startswith(
                "/"
            ):
                opts.data.files[mode][domain] = str(
                    Path(opts.data.files.base) / opts.data.files[mode][domain]
                )
            assert Path(
                opts.data.files[mode][domain]
            ).exists(), "Cannot find {}".format(str(opts.data.files[mode][domain]))

    return opts


def load_test_opts(test_file_path: str = "config/trainer/local_tests.yaml") -> Dict:
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


def get_git_revision_hash() -> str:
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception as e:
        return str(e)


def get_git_branch() -> str:
    """Get current git branch name

    Returns:
        str: git branch name
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
    except Exception as e:
        return str(e)


def kill_job(id: Union[int, str]) -> None:
    subprocess.check_output(["scancel", str(id)])


def write_hash(path: Union[str, Path]) -> None:
    hash_code = get_git_revision_hash()
    with open(path, "w") as f:
        f.write(hash_code)


def shortuid():
    return str(uuid4()).split("-")[0]


def datenowshort():
    """
    >>> a = str(datetime.datetime.now())
    >>> print(a)
    '2021-02-25 11:34:50.188072'
    >>> print(a[5:].split(".")[0].replace(" ", "_"))
    '02-25_11:35:41'

    Returns:
        str: month-day_h:m:s
    """
    return str(datetime.datetime.now())[5:].split(".")[0].replace(" ", "_")


def get_increased_path(path: Union[str, Path], use_date: bool = False) -> Path:
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
    fp = resolve(path)
    if not fp.exists():
        return fp

    if fp.is_file():
        if not use_date:
            while fp.exists():
                fp = fp.parent / f"{fp.stem}--{shortuid()}{fp.suffix}"
            return fp
        else:
            while fp.exists():
                time.sleep(0.5)
                fp = fp.parent / f"{fp.stem}--{datenowshort()}{fp.suffix}"
            return fp

    if not use_date:
        while fp.exists():
            fp = fp.parent / f"{fp.name}--{shortuid()}"
        return fp
    else:
        while fp.exists():
            time.sleep(0.5)
            fp = fp.parent / f"{fp.name}--{datenowshort()}"
        return fp

    # vals = []
    # for n in fp.parent.glob("{}*".format(fp.stem)):
    #     if re.match(r".+\(\d+\)", str(n.name)) is not None:
    #         name = str(n.name)
    #         start = name.index("(")
    #         end = name.index(")")
    #         vals.append(int(name[start + 1 : end]))
    # if vals:
    #     ext = " ({})".format(max(vals) + 1)
    # elif fp.exists():
    #     ext = " (1)"
    # else:
    #     ext = ""
    # return fp.parent / (fp.stem + ext + fp.suffix)


def env_to_path(path: str) -> str:
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


def flatten_opts(opts: Dict) -> dict:
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
                if v and isinstance(v[0], (Dict, dict)):
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


def get_comet_rest_api_key(
    path_to_config_file: Optional[Union[str, Path]] = None
) -> str:
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
        p = resolve(path_to_config_file)
    else:
        p = Path() / ".comet.config"
        if not p.exists():
            p = Path.home() / ".comet.config"
            if not p.exists():
                raise ValueError("Unable to find your COMET_REST_API_KEY")
    with p.open("r") as f:
        for keys in f:
            if "rest_api_key" in keys:
                return keys.strip().split("=")[-1].strip()
    raise ValueError("Unable to find your COMET_REST_API_KEY in {}".format(str(p)))


def get_files(dirName: str) -> list:
    # create a list of file and sub directories
    files = sorted(os.listdir(dirName))
    all_files = list()
    for entry in files:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            all_files = all_files + get_files(fullPath)
        else:
            all_files.append(fullPath)

    return all_files


def make_json_file(
    tasks: List[str],
    addresses: List[str],  # for windows user, use "\\" instead of using "/"
    json_names: List[str] = ["train_jsonfile.json", "val_jsonfile.json"],
    splitter: str = "/",
    pourcentage_val: float = 0.15,
) -> None:
    """
        How to use it?
    e.g.
    make_json_file(['x','m','d'], [
    '/network/tmp1/ccai/data/munit_dataset/trainA_size_1200/',
    '/network/tmp1/ccai/data/munit_dataset/seg_trainA_size_1200/',
    '/network/tmp1/ccai/data/munit_dataset/trainA_megadepth_resized/'
    ], ["train_r.json", "val_r.json"])

    Args:
        tasks (list): the list of image type like 'x', 'm', 'd', etc.
        addresses (list): the list of the corresponding address of the
            image type mentioned in tasks
        json_names (list): names for the json files, train being first
            (e.g. : ["train_r.json", "val_r.json"])
        splitter (str, optional): The path separator for the current OS.
            Defaults to '/'.
        pourcentage_val: pourcentage of files to go in validation set
    """
    assert len(tasks) == len(addresses), "keys and addresses must have the same length!"

    files = [get_files(addresses[j]) for j in range(len(tasks))]
    n_files_val = int(pourcentage_val * len(files[0]))
    n_files_train = len(files[0]) - n_files_val
    filenames = [files[0][:n_files_train], files[0][-n_files_val:]]

    file_address_map = {
        tasks[j]: {
            ".".join(file.split(splitter)[-1].split(".")[:-1]): file
            for file in files[j]
        }
        for j in range(len(tasks))
    }
    # The tasks of the file_address_map are like 'x', 'm', 'd'...
    # The values of the file_address_map are a dictionary whose tasks are the
    # filenames without extension whose values are the path of the filename
    # e.g. file_address_map =
    # {'x': {'A': 'path/to/trainA_size_1200/A.png', ...},
    #  'm': {'A': 'path/to/seg_trainA_size_1200/A.jpg',...}
    #  'd': {'A': 'path/to/trainA_megadepth_resized/A.bmp',...}
    # ...}

    for i, json_name in enumerate(json_names):
        dicts = []
        for j in range(len(filenames[i])):
            file = filenames[i][j]
            filename = file.split(splitter)[-1]  # the filename with 'x' extension
            filename_ = ".".join(
                filename.split(".")[:-1]
            )  # the filename without extension
            tmp_dict = {}
            for k in range(len(tasks)):
                tmp_dict[tasks[k]] = file_address_map[tasks[k]][filename_]
            dicts.append(tmp_dict)
        with open(json_name, "w", encoding="utf-8") as outfile:
            json.dump(dicts, outfile, ensure_ascii=False)


def append_task_to_json(
    path_to_json: Union[str, Path],
    path_to_new_json: Union[str, Path],
    path_to_new_images_dir: Union[str, Path],
    new_task_name: str,
):
    """Add all files for a task to an existing json file by creating a new json file
    in the specified path.
    Assumes that the files for the new task have exactly the same names as the ones
    for the other tasks

    Args:
        path_to_json: complete path to the json file to modify
        path_to_new_json: complete path to the new json file to be created
        path_to_new_images_dir: complete path of the directory where to find the
            images for the new task
        new_task_name: name of the new task

    e.g:
        append_json(
            "/network/tmp1/ccai/data/climategan/seg/train_r.json",
            "/network/tmp1/ccai/data/climategan/seg/train_r_new.json"
            "/network/tmp1/ccai/data/munit_dataset/trainA_seg_HRNet/unity_labels",
            "s",
        )
    """
    ims_list = None
    if path_to_json:
        path_to_json = Path(path_to_json).resolve()
        with open(path_to_json, "r") as f:
            ims_list = json.load(f)

    files = get_files(path_to_new_images_dir)

    if ims_list is None:
        raise ValueError(f"Could not find the list in {path_to_json}")

    new_ims_list = [None] * len(ims_list)
    for i, im_dict in enumerate(ims_list):
        new_ims_list[i] = {}
        for task, path in im_dict.items():
            new_ims_list[i][task] = path

    for i, im_dict in enumerate(ims_list):
        for task, path in im_dict.items():
            file_name = os.path.splitext(path)[0]  # removes extension
            file_name = file_name.rsplit("/", 1)[-1]  # only the file_name
            file_found = False
            for file_path in files:
                if file_name in file_path:
                    file_found = True
                    new_ims_list[i][new_task_name] = file_path
                    break
            if file_found:
                break
            else:
                print("Error! File ", file_name, "not found in directory!")
                return

    with open(path_to_new_json, "w", encoding="utf-8") as f:
        json.dump(new_ims_list, f, ensure_ascii=False)


def sum_dict(dict1: Union[dict, Dict], dict2: Union[Dict, dict]) -> Union[dict, Dict]:
    """Add dict2 into dict1"""
    for k, v in dict2.items():
        if not isinstance(v, dict):
            dict1[k] += v
        else:
            sum_dict(dict1[k], dict2[k])
    return dict1


def div_dict(dict1: Union[dict, Dict], div_by: float) -> dict:
    """Divide elements of dict1 by div_by"""
    for k, v in dict1.items():
        if not isinstance(v, dict):
            dict1[k] /= div_by
        else:
            div_dict(dict1[k], div_by)
    return dict1


def comet_id_from_url(url: str) -> Optional[str]:
    """
    Get comet exp id from its url:
    https://www.comet.ml/vict0rsch/climategan/2a1a4a96afe848218c58ac4e47c5375f
    -> 2a1a4a96afe848218c58ac4e47c5375f

    Args:
        url (str): comet exp url

    Returns:
        str: comet exp id
    """
    try:
        ids = url.split("/")
        ids = [i for i in ids if i]
        return ids[-1]
    except Exception:
        return None


@contextlib.contextmanager
def temp_np_seed(seed: Optional[int]) -> None:
    """
    Set temporary numpy seed:
    with temp_np_seed(123):
        np.random.permutation(3)

    Args:
        seed (int): temporary numpy seed
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_display_indices(opts: Dict, domain: str, length: int) -> list:
    """
    Compute the index of images to use for comet logging:
    if opts.comet.display_indices is an int, and domain is real:
        return range(int)
    if opts.comet.display_indices is an int, and domain is sim:
        return permutation(length)[:int]
    if opts.comet.display_indices is a list:
        return list

    otherwise return []


    Args:
        opts (addict.Dict): options
        domain (str): domain for those indices
        length (int): length of dataset for the permutation

    Returns:
        list(int): The indices to display
    """
    if domain == "rf":
        dsize = max([opts.comet.display_size, opts.train.fid.get("n_images", 0)])
    else:
        dsize = opts.comet.display_size
    if dsize > length:
        print(
            f"Warning: dataset is smaller ({length} images) "
            + f"than required display indices ({dsize})."
            + f" Selecting {length} images."
        )

    display_indices = []
    assert isinstance(dsize, (int, list)), "Unknown display size {}".format(dsize)
    if isinstance(dsize, int):
        assert dsize >= 0, "Display size cannot be < 0"
        with temp_np_seed(123):
            display_indices = list(np.random.permutation(length)[:dsize])
    elif isinstance(dsize, list):
        display_indices = dsize

    if not display_indices:
        print("Warning: no display indices (utils.get_display_indices)")

    return display_indices


def get_latest_path(path: Union[str, Path]) -> Path:
    """
    Get the file/dir with largest increment i as `file (i).ext`

    Args:
        path (str or pathlib.Path): base pattern

    Returns:
        Path: path found
    """
    p = Path(path).resolve()
    s = p.stem
    e = p.suffix
    files = list(p.parent.glob(f"{s}*(*){e}"))
    indices = list(p.parent.glob(f"{s}*(*){e}"))
    indices = list(map(lambda f: f.name, indices))
    indices = list(map(lambda x: re.findall(r"\((.*?)\)", x)[-1], indices))
    indices = list(map(int, indices))
    if not indices:
        f = p
    else:
        f = files[np.argmax(indices)]
    return f


def get_existing_jobID(output_path: Path) -> str:
    """
    If the opts in output_path have a jobID, return it. Else, return None

    Args:
        output_path (pathlib.Path | str): where to  look

    Returns:
        str | None: jobid
    """
    op = Path(output_path)
    if not op.exists():
        return

    opts_path = get_latest_path(op / "opts.yaml")

    if not opts_path.exists():
        return

    with opts_path.open("r") as f:
        opts = yaml.safe_load(f)

    jobID = opts.get("jobID", None)

    return jobID


def find_existing_training(opts: Dict) -> Optional[Path]:
    """
    Looks in all directories like output_path.parent.glob(output_path.name*)
    and compares the logged slurm job id with the current opts.jobID

    If a match is found, the training should automatically continue in the
    matching output directory

    If no match is found, this is a new job and it should have a new output path

    Args:
        opts (Dict): trainer's options

    Returns:
        Optional[Path]: a path if a matchin jobID is found, None otherwise
    """
    if opts.jobID is None:
        print("WARNING: current JOBID is None")
        return

    print("---------- Current job id:", opts.jobID)

    path = Path(opts.output_path).resolve()
    parent = path.parent
    name = path.name

    try:
        similar_dirs = [p.resolve() for p in parent.glob(f"{name}*") if p.is_dir()]

        for sd in similar_dirs:
            candidate_jobID = get_existing_jobID(sd)
            if candidate_jobID is not None and str(opts.jobID) == str(candidate_jobID):
                print(f"Found matching job id in {sd}\n")
                return sd
        print("Did not find a matching job id in \n {}\n".format(str(similar_dirs)))
    except Exception as e:
        print("ERROR: Could not resume (find_existing_training)", e)


def pprint(*args: List[Any]):
    """
    Prints *args within a box of "=" characters
    """
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


def get_existing_comet_id(path: str) -> Optional[str]:
    """
    Returns the id of the existing comet experiment stored in path

    Args:
        path (str): Output pat where to look for the comet exp

    Returns:
        Optional[str]: comet exp's ID if any was found
    """
    comet_previous_path = get_latest_path(Path(path) / "comet_url.txt")
    if comet_previous_path.exists():
        with comet_previous_path.open("r") as f:
            url = f.read().strip()
            return comet_id_from_url(url)


def get_latest_opts(path):
    """
    get latest opts dumped in path if they look like *opts*.yaml
    and were increased as
    opts.yaml < opts (1).yaml < opts (2).yaml etc.

    Args:
        path (str or pathlib.Path): where to look for opts

    Raises:
        ValueError: If no match for *opts*.yaml is found

    Returns:
        addict.Dict: loaded opts
    """
    path = Path(path)
    opts = get_latest_path(path / "opts.yaml")
    assert opts.exists()
    with opts.open("r") as f:
        opts = Dict(yaml.safe_load(f))

    events_path = Path(__file__).parent.parent / "shared" / "trainer" / "events.yaml"
    if events_path.exists():
        with events_path.open("r") as f:
            events_dict = yaml.safe_load(f)
        events_dict = Dict(events_dict)
        opts.events = events_dict

    return opts


def text_to_array(text, width=640, height=40):
    """
    Creates a numpy array of shape height x width x 3 with
    text written on it using PIL

    Args:
        text (str): text to write
        width (int, optional): Width of the resulting array. Defaults to 640.
        height (int, optional): Height of the resulting array. Defaults to 40.

    Returns:
        np.ndarray: Centered text
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        font = ImageFont.truetype("UnBatang.ttf", 25)
    except OSError:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text)
    h = 40 // 2 - 3 * text_height // 2
    w = width // 2 - text_width
    d.text((w, h), text, font=font, fill=(30, 30, 30))
    return np.array(img)


def all_texts_to_array(texts, width=640, height=40):
    """
    Creates an array of texts, each of height and width specified
    by the args, concatenated along their width dimension

    Args:
        texts (list(str)): List of texts to concatenate
        width (int, optional): Individual text's width. Defaults to 640.
        height (int, optional): Individual text's height. Defaults to 40.

    Returns:
        list: len(texts) text arrays with dims height x width x 3
    """
    return [text_to_array(text, width, height) for text in texts]


class Timer:
    def __init__(self, name="", store=None, precision=3, ignore=False, cuda=True):
        self.name = name
        self.store = store
        self.precision = precision
        self.ignore = ignore
        self.cuda = cuda

        if cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def format(self, n):
        return f"{n:.{self.precision}f}"

    def __enter__(self):
        """Start a new timer as a context manager"""
        if self.cuda:
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        if self.ignore:
            return

        if self.cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            new_time = self._start_event.elapsed_time(self._end_event) / 1000
        else:
            t = time.perf_counter()
            new_time = t - self._start_time

        if self.store is not None:
            assert isinstance(self.store, list)
            self.store.append(new_time)
        if self.name:
            print(f"[{self.name}] Elapsed time: {self.format(new_time)}")


def get_loader_output_shape_from_opts(opts):
    transforms = opts.data.transforms

    t = None
    for t in transforms[::-1]:
        if t.name == "resize":
            break
    assert t is not None

    if isinstance(t.new_size, Dict):
        return {
            task: (
                t.new_size.get(task, t.new_size.default),
                t.new_size.get(task, t.new_size.default),
            )
            for task in opts.tasks + ["x"]
        }
    assert isinstance(t.new_size, int)
    new_size = (t.new_size, t.new_size)
    return {task: new_size for task in opts.tasks + ["x"]}


def find_target_size(opts, task):
    target_size = None
    if isinstance(opts.data.transforms[-1].new_size, int):
        target_size = opts.data.transforms[-1].new_size
    else:
        if task in opts.data.transforms[-1].new_size:
            target_size = opts.data.transforms[-1].new_size[task]
        else:
            assert "default" in opts.data.transforms[-1].new_size
            target_size = opts.data.transforms[-1].new_size["default"]

    return target_size


def to_128(im, w_target=-1):
    h, w = im.shape[:2]
    aspect_ratio = h / w
    if w_target < 0:
        w_target = w

    nw = int(w_target / 128) * 128
    nh = int(nw * aspect_ratio / 128) * 128

    return nh, nw


def is_image_file(filename):
    """Check that a file's name points to a known image format"""
    if isinstance(filename, Path):
        return filename.suffix in IMG_EXTENSIONS

    return Path(filename).suffix in IMG_EXTENSIONS


def find_images(path, recursive=False):
    """
    Get a list of all images contained in a directory:

    - path.glob("*") if not recursive
    - path.glob("**/*") if recursive
    """
    p = Path(path)
    assert p.exists()
    assert p.is_dir()
    pattern = "*"
    if recursive:
        pattern += "*/*"

    return [i for i in p.glob(pattern) if i.is_file() and is_image_file(i)]


def cols():
    try:
        col = os.get_terminal_size().columns
    except Exception:
        col = 50
    return col


def upload_images_to_exp(
    path, exp=None, project_name="climategan-eval", sleep=-1, verbose=0
):
    ims = find_images(path)
    end = None
    c = cols()
    if verbose == 1:
        end = "\r"
    if verbose > 1:
        end = "\n"
    if exp is None:
        exp = Experiment(project_name=project_name)
    for im in ims:
        exp.log_image(str(im))
        if verbose > 0:
            if verbose == 1:
                print(" " * (c - 1), end="\r", flush=True)
            print(str(im), end=end, flush=True)
        if sleep > 0:
            time.sleep(sleep)
    return exp
