"""Tensor-utils
"""
from pathlib import Path

# from copy import copy
from threading import Thread

import numpy as np
import torch
from skimage import io as skio
from torch.nn import init


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

    mapping = {"r": 0, "s": 1}

    if not all(domain in mapping for domain in domains):
        raise ValueError("Unknown domains {} should be in {}".format(domains, list(mapping.keys())))

    target = torch.tensor([mapping[domain] for domain in domains])

    if one_hot:
        one_hot_target = torch.FloatTensor(len(target), 2)  # 4 domains
        one_hot_target.zero_()
        one_hot_target.scatter_(1, target.unsqueeze(1), 1)
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        target = one_hot_target
    return target


def fake_domains_to_class_tensor(domains, one_hot=False):
    """Converts a list of strings to a 1D Tensor representing the fake domains
    (real or sim only)

    fake_domains_to_class_tensor(["s", "r"], False)
    >>> torch.Tensor([0, 2])


    Args:
        domain (list(str)): each element of the list should be in {r, s}
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
        target = torch.FloatTensor(len(domains), 2)
        target.fill_(0.25)

    else:
        mapping = {"r": 1, "s": 0}

        if not all(domain in mapping for domain in domains):
            raise ValueError(
                "Unknown domains {} should be in {}".format(domains, list(mapping.keys()))
            )

        target = torch.tensor([mapping[domain] for domain in domains])
    return target


def show_tanh_tensor(tensor):
    import skimage

    if isinstance(tensor, torch.Tensor):
        image = tensor.permute(1, 2, 0).detach().numpy()
    else:
        if tensor.shape[-1] != 3:
            image = tensor.transpose(1, 2, 0)

    if image.min() < 0 and image.min() > -1:
        image = image / 2 + 0.5
    elif image.min() < -1:
        raise ValueError("can't handle this data")

    skimage.io.imshow(image)


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
    bit = torch.ones(shape[0], probs.shape[-1], *shape[-2:]).to(torch.float32).to(probs.device)
    bit *= probs[None, :, None, None].to(torch.float32)
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


def get_conditioning_tensor(x, task_tensors, classifier_probs=None):
    """creates the 4D tensor to condition the translation on by concatenating d, h, s, w
    and an optional conditioning bit:

    Args:
        x (torch.Tensor): tensor whose shape we'll use to expand the bit
        task_tensors (torch.Tensor): dictionnary task: conditioning tensor
        classifier_probs (list, optional): 1-hot encoded depending on the
            domain to use. Defaults to None.

    Returns:
        torch.Tensor: conditioning tensor, all tensors concatenated
            on the channel dim
    """

    K = [v for k, v in sorted(task_tensors.items(), key=lambda t: t[0])]

    assert all(len(t.shape) == 4 for t in K)

    if classifier_probs is None:
        return torch.cat(K, dim=1)

    bit = get_4D_bit(x.shape, classifier_probs).detach().to(x.device)
    # bit => batchsize * conditioning tensor
    # conditioning tensor => 4 x h x d, with 0s or 1s as classifier_probs
    return torch.cat(K + [bit], dim=1)


def print_net(net):
    if hasattr(net, "model"):
        for b in net.model:
            name = b.__class__.__name__
            if "Conv2dBlock" in name:
                print(f"{name}: {b.weight.shape}")


def slice_batch(batch, slice_size):
    assert slice_size > 0
    for k, v in batch.items():
        if isinstance(v, dict):
            for task, d in v.items():
                batch[k][task] = d[:slice_size]
        else:
            batch[k] = v[:slice_size]
    return batch


def save_tanh_tensor(image, path):
    """Save an image which can be numpy or tensor, 2 or 3 dims (no batch)
    to path.

    Args:
        image (np.array or torch.Tensor): image to save
        path (pathlib.Path or str): where to save the image
    """
    path = Path(path)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[-1] != 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    if image.min() < 0 and image.min() > -1:
        image = image / 2 + 0.5
    elif image.min() < -1:
        image -= image.min()
        image /= image.max()
        # print("Warning: scaling image data in save_tanh_tensor")

    skio.imsave(path, (image * 255).astype(np.uint8))


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
                images_to_save["paths"] += [root / "im_{}_{}_{}.png".format(step, domain, imid)]
                images_to_save["images"].append(im)
    if num_threads > 0:
        threaded_write(images_to_save["images"], images_to_save["paths"], num_threads)
    else:
        for im, path in zip(images_to_save["images"], images_to_save["paths"]):
            save_tanh_tensor(im, path)


def threaded_write(images, paths, num_threads=5):
    t_im = []
    t_p = []
    for im, p in zip(images, paths):
        t_im.append(im)
        t_p.append(p)
        if len(t_im) == num_threads:
            ts = [Thread(target=save_tanh_tensor, args=(_i, _p)) for _i, _p in zip(t_im, t_p)]
            list(map(lambda t: t.start(), ts))
            list(map(lambda t: t.join(), ts))
            t_im = []
            t_p = []
    if t_im:
        ts = [Thread(target=save_tanh_tensor, args=(_i, _p)) for _i, _p in zip(t_im, t_p)]
        list(map(lambda t: t.start(), ts))
        list(map(lambda t: t.join(), ts))
