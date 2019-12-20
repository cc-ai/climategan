import os
import re
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from copy import copy
import yaml
from addict import Dict
from torch.nn import init
import torch


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


def load_opts(path, default=None):
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
        [type]: [description]
    """
    if default is None:
        return load_default_opts(path)

    default_opts = load_default_opts(default)
    with open(path, "r") as f:
        overriding_opts = Dict(yaml.safe_load(f))
    default_opts.update(overriding_opts)

    return default_opts


def load_default_opts(path):
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

            # set default decoder parameters for all tasks
            for k in opts.tasks:
                tmp = copy(opts.gen.default)
                if k in opts.gen:
                    tmp.update(opts.gen[k])
                opts.gen[k] = tmp

            # set default discriminator parameters
            for k in {"a", "t"} & set(opts.tasks):
                tmp = copy(opts.dis.default)
                if k in opts.dis:
                    tmp.update(opts.dis[k])
                opts.dis[k] = tmp

            # set default loss coefficients for tasks and auto-encoding
            default = opts.train.lambdas.default
            for k in opts.tasks:
                if k not in opts.train.lambdas.tasks:
                    opts.train.lambdas.G.tasks[k] = default
            if "a" not in opts.train.lambdas.auto:
                opts.train.lambdas.auto.a = default
            if "a" not in opts.train.lambdas.gan:
                opts.train.lambdas.gan.a = default
            if "a" not in opts.train.lambdas.cycle:
                opts.train.lambdas.cycle.a = default

            if "t" not in opts.train.lambdas.auto:
                opts.train.lambdas.auto.t = default
            if "t" not in opts.train.lambdas.gan:
                opts.train.lambdas.gan.t = default
            if "t" not in opts.train.lambdas.cycle:
                opts.train.lambdas.cycle.t = default

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


def init_weights(net, init_type="normal", init_gain=0.02, verbose=0):
    """Initialize network weights.
        Parameters:
            net (network)     -- network to be initialized
            init_type (str)   -- the name of an initialization method:
                                 normal | xavier | kaiming | orthogonal
            init_gain (float) -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper.
        But xavier and kaiming might work better for some applications.
        Feel free to try yourself.
        """

    if not init_type:
        print("init_type is {}, defaulting to normal".format(init_type))
        init_type = "normal"
    if not init_gain:
        print("init_gain is {}, defaulting to 0.02".format(init_gain))
        init_gain = 0.02

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if verbose > 0:
        print("initialize %s with %s" % (net.__class__.__name__, init_type))
    net.apply(init_func)


def freeze(self, net):
    """Sets requires_grad = False to all the net's parameters

    Args:
        net (nn.Module): Network to freeze
    """
    for p in net.parameters():
        p.requires_grad = False


def domains_to_class_tensor(domains, loss):
    """Converts a list of strings to a 1D Tensor representing the domains

    domain_to_class_tensor(["sf", "rn"])
    >>> torch.Tensor([2, 1])


    Args:
        domain (list(str)): each element of the list should be in {rf, rn, sf, sn}
        loss (str): loss to use according to the config file
    Raises:
        ValueError: One of the domains listed is not in {rf, rn, sf, sn}

    Returns:
        torch.Tensor: 1D tensor mapping a domain to an int (not 1-hot) if loss is CE
        or 2D tensor mapping a domain to an int (one-hot) if loss is L1 or L2
    """

    mapping = {"rf": 0, "rn": 1, "sf": 2, "sn": 3}

    if not all(domain in mapping for domain in domains):
        raise ValueError(
            "Unknown domains {} should be in {}".format(domains, list(mapping.keys()))
        )

    target = torch.tensor([mapping[domain] for domain in domains])

    if (loss == "l1") or (loss == "l2"):
        one_hot_target = torch.FloatTensor(len(target), 4)  # 4 domains
        one_hot_target.zero_()
        one_hot_target.scatter_(1, target.unsqueeze(1), 1)
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        target = one_hot_target
    return target


def fake_domains_to_class_tensor(domains, loss):
    """Converts a list of strings to a 1D Tensor representing the fake domains
    (real or sim only)

    domain_to_class_tensor(["sf", "rn"])
    >>> torch.Tensor([0, 3])


    Args:
        domain (list(str)): each element of the list should be in {rf, rn, sf, sn}
    Raises:
        ValueError: One of the domains listed is not in {rf, rn, sf, sn}

    Returns:
        torch.Tensor: 1D tensor mapping a domain to an int (not 1-hot) if loss is CE 
        or a 2D tensor filled with 0.5 to fool the 
    """
    if (loss == "l1") or (loss == "l2"):
        target = torch.FloatTensor(len(domains), 4)
        target.fill_(0.25)

    else:
        mapping = {"rf": 2, "rn": 3, "sf": 0, "sn": 1}

        if not all(domain in mapping for domain in domains):
            raise ValueError(
                "Unknown domains {} should be in {}".format(
                    domains, list(mapping.keys())
                )
            )

        target = torch.tensor([mapping[domain] for domain in domains])
    return target

