"""All non-tensor utils
"""
import os
import re
import subprocess
import json
from pathlib import Path
import shutil
import yaml
from addict import Dict
import contextlib
import numpy as np
from typing import Union, Optional, List, Any
import traceback

comet_kwargs = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}


def copy_run_files(opts: Dict) -> None:
    """
    Copy the opts's sbatch_file to output_path

    Args:
        opts (addict.Dict): options
    """
    if opts.sbatch_file:
        p = Path(opts.sbatch_file)
        if p.exists():
            o = Path(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)
    if opts.exp_file:
        p = Path(opts.exp_file)
        if p.exists():
            o = Path(opts.output_path)
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
    # TODO add assert: if deeplabv2 then res_dim = 2048
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
    assert default or path

    if path:
        path = Path(path).resolve()

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

    opts.domains = []
    if "m" in opts.tasks or "s" in opts.tasks:
        opts.domains.extend(["r", "s"])
    if "p" in opts.tasks:
        opts.domains.append("rf")
    opts.domains = list(set(opts.domains))

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
            opts.data.files[mode][domain] = str(
                Path(opts.data.files.base) / opts.data.files[mode][domain]
            )

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


def kill_job(id: Union[int, str]) -> None:
    subprocess.check_output(["scancel", str(id)])


def write_hash(path: Union[str, Path]) -> None:
    hash_code = get_git_revision_hash()
    with open(path, "w") as f:
        f.write(hash_code)


def get_increased_path(path: Union[str, Path]) -> Path:
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
    for n in fp.parent.glob("{}*".format(fp.stem)):
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

    return fp.parent / (fp.stem + ext + fp.suffix)


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
            "/network/tmp1/ccai/data/omnigan/seg/train_r.json",
            "/network/tmp1/ccai/data/omnigan/seg/train_r_new.json"
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
    """Add dict2 into dict1
    """
    for k, v in dict2.items():
        if not isinstance(v, dict):
            dict1[k] += v
        else:
            sum_dict(dict1[k], dict2[k])
    return dict1


def div_dict(dict1: Union[dict, Dict], div_by: float) -> dict:
    """Divide elements of dict1 by div_by
    """
    for k, v in dict1.items():
        if not isinstance(v, dict):
            dict1[k] /= div_by
        else:
            div_dict(dict1[k], div_by)
    return dict1


def comet_id_from_url(url: str) -> Optional[str]:
    """
    Get comet exp id from its url:
    https://www.comet.ml/vict0rsch/omnigan/2a1a4a96afe848218c58ac4e47c5375f
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
    indices = list(map(lambda x: re.findall("\((.*?)\)", x)[-1], indices))
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

    print("Current job id:", opts.jobID)

    path = Path(opts.output_path).resolve()
    parent = path.parent
    name = path.name

    try:
        similar_dirs = [p.resolve() for p in parent.glob(f"{name}*") if p.is_dir()]

        for sd in similar_dirs:
            candidate_jobID = get_existing_jobID(sd)
            if candidate_jobID is not None and opts.jobID == candidate_jobID:
                print(f"Found matching job id in {sd}")
                return sd
        print("Did not find a matching job id")
    except Exception as e:
        print("Could not resume (find_existing_training)", e)


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
        return Dict(yaml.safe_load(f))
