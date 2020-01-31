import os
import re
import subprocess
from copy import copy
from pathlib import Path

from skimage import io as skio
import torch
import yaml
from addict import Dict
from torch.nn import init
import numpy as np
from threading import Thread


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
        addict.Dict: options dictionnary, with overwritten default values
    """
    if default is None:
        default_opts = Dict()
    else:
        with open(default, "r") as f:
            default_opts = Dict(yaml.safe_load(f))

    with open(path, "r") as f:
        overriding_opts = Dict(yaml.safe_load(f))

    default_opts.update(overriding_opts)

    return set_opts_values(default_opts)


def set_opts_values(opts):
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
        opts (addict.Dict): Options to set up the default values for

    Returns:
        addict.Dict: the configuration object
    """

    for mode in ["train", "val"]:
        for domain in opts.data.files[mode]:
            opts.data.files[mode][domain] = str(
                Path(opts.data.files.base) / opts.data.files[mode][domain]
            )

    # set default decoder parameters for all tasks
    for k in opts.tasks + ["encoder"]:
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


def freeze(net):
    """Sets requires_grad = False to all the net's parameters

    Args:
        net (nn.Module): Network to freeze
    """
    for p in net.parameters():
        p.requires_grad = False


def domains_to_class_tensor(domains, one_hot=False):
    """Converts a list of strings to a 1D Tensor representing the domains

    domains_to_class_tensor(["sf", "rn"])
    >>> torch.Tensor([2, 1])

    Args:
        domain (list(str)): each element of the list should be in {rf, rn, sf, sn}
        one_hot (bool, optional): whether or not to 1-h encode class labels.
            Defaults to False.
    Raises:
        ValueError: One of the domains listed is not in {rf, rn, sf, sn}

    Returns:
        torch.Tensor: 1D tensor mapping a domain to an int (not 1-hot) or 1-hot
            domain labels in a 2D tensor
    """

    mapping = {"rf": 0, "rn": 1, "sf": 2, "sn": 3}

    if not all(domain in mapping for domain in domains):
        raise ValueError(
            "Unknown domains {} should be in {}".format(domains, list(mapping.keys()))
        )

    target = torch.tensor([mapping[domain] for domain in domains])

    if one_hot:
        one_hot_target = torch.FloatTensor(len(target), 4)  # 4 domains
        one_hot_target.zero_()
        one_hot_target.scatter_(1, target.unsqueeze(1), 1)
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        target = one_hot_target
    return target


def fake_domains_to_class_tensor(domains, one_hot=False):
    """Converts a list of strings to a 1D Tensor representing the fake domains
    (real or sim only)

    fake_domains_to_class_tensor(["sf", "rn"], False)
    >>> torch.Tensor([0, 3])


    Args:
        domain (list(str)): each element of the list should be in {rf, rn, sf, sn}
        one_hot (bool, optional): whether or not to 1-h encode class labels.
            Defaults to False.
    Raises:
        ValueError: One of the domains listed is not in {rf, rn, sf, sn}

    Returns:
        torch.Tensor: 1D tensor mapping a domain to an int (not 1-hot) or
            a 2D tensor filled with 0.25 to fool the classifier (equiprobability
            for each domain).
    """
    if one_hot:
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
