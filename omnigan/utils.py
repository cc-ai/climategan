"""All non-tensor utils
"""
import os
import re
import subprocess
from copy import copy
from pathlib import Path

from skimage import io as skio
import torch
from addict import Dict
from torch.nn import init
import numpy as np
from threading import Thread
import yaml
from copy import deepcopy


def load_exp(path):
    path = Path(path)
    with open(path, "r") as f:
        xopts = Dict(yaml.safe_load(f))

    if xopts.experiment.repeat and xopts.experiment.repeat > 1:
        xopts.runs = [
            deepcopy(run) for run in xopts.runs for _ in range(xopts.experiment.repeat)
        ]

    for i, run in enumerate(xopts.runs):
        xopts.runs[i].trainer = resolve_sample(run.trainer)

    for i, run in enumerate(xopts.runs):
        conf = (
            env_to_path(xopts.experiment.config)
            if xopts.experiment.config_file
            else None
        )
        defaults = (
            env_to_path(xopts.experiment.defaults)
            if xopts.experiment.defaults
            else None
        )
        trainer_opts = load_opts(conf, defaults)
        trainer_opts.update(run.trainer)
        trainer_opts.output_path = str(
            Path(env_to_path(xopts.experiment.base_dir)) / "run"
        )
        xopts.runs[i].trainer = trainer_opts

    return xopts


def write_run_template(xopts, i, template_path, write_path):
    ropt = xopts.runs[i]
    exp = xopts.experiment
    with open(template_path, "r") as f:
        template = f.readlines()

    beluga = bool(os.environ.get("SCRATCH"))

    new_template = []
    for line in template:
        line = line.strip()
        if "{{" not in line:
            new_template.append(line)
            continue
        for pattern in re.findall(r"{{\w+}}", line):
            ignore = False
            param = pattern.replace("{{", "").replace("}}", "")
            replace = ""
            if param == "cpu":
                replace = str(ropt.sbatch.cpu)
            elif param == "gpu":
                replace = str(ropt.sbatch.gpu)
            elif param == "mem":
                replace = str(ropt.sbatch.mem)
            elif param == "time":
                if not beluga:
                    ignore = True
                replace = str(ropt.sbatch.duration)
            elif param == "output_path":
                replace = str(ropt.trainer.output_path)
            elif param == "main_partition":
                if beluga:
                    ignore = True
                replace = f"#SBATCH -p {ropt.sbatch.partition}"
            elif param == "config":
                replace = "--config=" + str(
                    Path(ropt.trainer.output_path) / "config.yaml"
                )
            elif param == "no_comet":
                replace = "--no_comet" if ropt.experiment.no_comet else ""
            elif param == "exp_desc":
                replace = f'--exp_desc="{str(exp.exp_desc)}"' if exp.exp_desc else ""
            elif param == "dev_mode":
                replace = "--dev_mode" if exp.dev_mode else ""
            elif param == "note":
                replace = f'--note="{str(ropt.comet.note)}"' if ropt.comet.note else ""
            elif param == "tags":
                replace = ""
                for t in ropt.comet.tags:
                    replace += f"--tag {'_'.join(t.split())} "
            else:
                ignore = True
            line = re.sub(pattern, replace, line)
        if not ignore:
            new_template.append(line)

    with open(write_path, "w") as f:
        f.write("\n".join(new_template))


def resolve_sample(dic):
    for k, v in dic.items():
        if isinstance(v, Dict):
            if "sample" in v:
                dic[k] = sample_param(v)
            else:
                dic[k] = resolve_sample(v)
    return dic


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
    if "sample" not in sample_dict:
        return sample_dict
    if sample_dict["sample"] == "range":
        value = np.random.choice(np.arange(*sample_dict["from"]))
    elif sample_dict["sample"] == "list":
        value = np.random.choice(sample_dict["from"])
    elif sample_dict["sample"] == "uniform":
        value = np.random.uniform(*sample_dict["from"])
    else:
        raise ValueError("Unknown sample type in dict " + str(sample_dict))
    return value


def load_opts(path=None, default=None):
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


def show_tanh_tensor(image):
    """Show an image which can be numpy or tensor, 2 or 3 dims (no batch)

    Args:
        image (np.array or torch.Tensor): image to save

    Raises:
        ValueError: If data.min() < -1
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().numpy()
        if image.shape[-1] != 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

    if image.min() < 0 and image.min() > -1:
        image = image / 2 + 0.5
    elif image.min() < -1:
        raise ValueError("can't handle this data")

    skio.imshow(image)


def save_batch(multi_domain_batch, root="./", step=0, num_threads=5):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images_to_save = {"paths": [], "images": []}
    for domain, batch in multi_domain_batch.items():
        y = batch["data"].get("y")
        x = batch["data"]["x"]
        if y is not None:
            paths = batch["paths"]["x"]
            imtensor = torch.cat([x, y], dim=-1)
            for i, im in enumerate(imtensor):
                imid = Path(paths[i]).stem[:10]
                images_to_save["paths"] += [
                    root / "im_{}_{}_{}.png".format(step, domain, imid)
                ]
                images_to_save["images"].append(im)
    if num_threads > 0:
        threaded_write(images_to_save["images"], images_to_save["paths"], num_threads)
    else:
        for im, path in zip(images_to_save["images"], images_to_save["paths"]):
            save_tanh_tensor(im, path)


def save_tanh_tensor(image, path):
    """Save an image which can be numpy or tensor, 2 or 3 dims (no batch)
    to path.

    Args:
        image (np.array or torch.Tensor): image to save
        path (pathlib.Path or str): where to save the image
    """
    path = Path(path)
    if isinstance(image, torch.Tensor):
        image = image.detach().numpy()
        if image.shape[-1] != 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    if image.min() < 0 and image.min() > -1:
        image = image / 2 + 0.5
    elif image.min() < -1:
        image -= image.min()
        image /= image.max()
        # print("Warning: scaling image data in save_tanh_tensor")

    skio.imsave(path, (image * 255).astype(np.uint8))


def get_4D_bit(shape, probs):
    """transforms domain probabilities (batch x # of domains)
    into a 4D tensor repeating probs as feature maps

    Args:
        shape (list): batch_size x (useless) x h x w
        probs (torch.Tensor): probabilities of belonging to a domain
            (batch x # of domains)

    Returns:
        torch.Tensor: batch x # of domains x h x w
    """
    probs = probs if isinstance(probs, torch.Tensor) else torch.tensor(probs)
    bit = torch.ones(*probs.shape, *shape[-2:])
    bit *= probs[:, :, None, None]
    return bit


def fake_batch(batch, fake):
    """create fake batch for the cycle reconstruction: copy all references in batch into
    cycle_batch BUT overwrite batch["data"]["x"] in order to save memory (instead of
    just deepcopy(batch) which would unnecessarily duplicate the rest of the data)

    Args:
        batch (dict): batch dictionnary with keys ['data', 'paths', 'domain', 'mode']
        fake (torch.Tensor): tensor which should replace batch["data"]["x"], for
            instance to use in the cycle reconstruction
    """
    return {**batch, **{"data": {**batch["data"], **{"x": fake}}}}


def threaded_write(images, paths, num_threads=5):
    t_im = []
    t_p = []
    for im, p in zip(images, paths):
        t_im.append(im)
        t_p.append(p)
        if len(t_im) == num_threads:
            ts = [
                Thread(target=save_tanh_tensor, args=(_i, _p))
                for _i, _p in zip(t_im, t_p)
            ]
            list(map(lambda t: t.start(), ts))
            list(map(lambda t: t.join(), ts))
            t_im = []
            t_p = []
    if t_im:
        ts = [
            Thread(target=save_tanh_tensor, args=(_i, _p)) for _i, _p in zip(t_im, t_p)
        ]
        list(map(lambda t: t.start(), ts))
        list(map(lambda t: t.join(), ts))


def slice_batch(batch, slice_size):
    assert slice_size > 0
    for k, v in batch.items():
        if isinstance(v, dict):
            for task, d in v.items():
                batch[k][task] = d[:slice_size]
        else:
            batch[k] = v[:slice_size]
    return batch


def decode_mega_depth(pred_log_depth, numpy=False):
    """Transforms the inference of a mega_depth model into an image:
    * torch.Tensor in [0, 1] as torch.float32 if numpy == False
    * else numpy.array in [0, 255] as np.uint8

    Args:
        pred_log_depth (torch.Tensor): inference on 1 image of mega_depth
        numpy (bool, optional): Whether to return a float tensor or an int array.
         Defaults to False.

    Returns:
        [torch.Tensor or numpy.array]: decoded depth
    """
    pred_depth = torch.exp(pred_log_depth)
    # visualize prediction using inverse depth, so that we don't need
    # sky segmentation (if you want to use RGB map for visualization,
    # you have to run semantic segmentation to mask the sky first
    # since the depth of sky is random from CNN)
    pred_inv_depth = 1 / pred_depth
    # you might also use percentile for better visualization
    max = pred_inv_depth.max(dim=-1, keepdim=True)[0]
    _max = pred_inv_depth.max(dim=-1, keepdim=False)[0]
    while len(_max.shape) != 1:
        max = max.max(dim=-1, keepdim=True)[0]
        _max = _max.max(dim=-1, keepdim=False)[0]
    pred_inv_depth = pred_inv_depth / max
    if numpy:
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        return (pred_inv_depth * 255).astype(np.uint8).squeeze()
    return pred_inv_depth


def shuffle_batch_tuple(mbt):
    """shuffle the order of domains in the batch

    Args:
        mbt (tuple): multi-batch tuple

    Returns:
        list: randomized list of domain-specific batches
    """
    assert isinstance(mbt, (tuple, list))
    assert len(mbt) > 0
    perm = np.random.permutation(len(mbt))
    return [mbt[i] for i in perm]
