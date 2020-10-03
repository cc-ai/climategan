"""Tensor-utils
"""
from pathlib import Path

# from copy import copy
from threading import Thread

import numpy as np
import torch
from torch.autograd import Variable
from skimage import io as skio
from torch.nn import init
import torch.nn as nn


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
        raise ValueError(
            "Unknown domains {} should be in {}".format(domains, list(mapping.keys()))
        )

    target = torch.tensor([mapping[domain] for domain in domains])

    if one_hot:
        one_hot_target = torch.FloatTensor(len(target), 2)  # 2 domains
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
        target.fill_(0.5)

    else:
        mapping = {"r": 1, "s": 0}

        if not all(domain in mapping for domain in domains):
            raise ValueError(
                "Unknown domains {} should be in {}".format(
                    domains, list(mapping.keys())
                )
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


def norm_tensor(t):
    t = t - torch.min(t)
    t /= torch.max(t)
    return t


def get_normalized_depth_t(arr, domain, normalize=False):
    if domain == "r":
        # megadepth depth
        arr = arr.unsqueeze(0)
        if normalize:
            arr = arr - torch.min(arr)
            arr = torch.true_divide(arr, torch.max(arr))
    elif domain == "s":
        # from 3-channel depth encoding from Unity simulator to 1-channel [0-1] values
        arr = decode_unity_depth_t(arr, log=False, normalize=normalize)
    return arr


def decode_unity_depth_t(unity_depth, log=True, normalize=False, numpy=False, far=1000):
    """Transforms the 3-channel encoded depth map from our Unity simulator to 1-channel depth map
    containing metric depth values.
    The depth is encoded in the following way:
    - The information from the simulator is (1 - LinearDepth (in [0,1])).
        far corresponds to the furthest distance to the camera included in the depth map.
        LinearDepth * far gives the real metric distance to the camera.
    - depth is first divided in 31 slices encoded in R channel with values ranging from 0 to 247
    - each slice is divided again in 31 slices, whose value is encoded in G channel
    - each of the G slices is divided into 256 slices, encoded in B channel
    In total, we have a discretization of depth into N = 31*31*256 - 1 possible values, covering a range of
    far/N meters.
    Note that, what we encode here is 1 - LinearDepth so that the furthest point is [0,0,0] (that is sky)
    and the closest point[255,255,255]
    The metric distance associated to a pixel whose depth is (R,G,B) is :
        d = (far/N) * [((255 - R)//8)*256*31 + ((255 - G)//8)*256 + (255 - B)]
    * torch.Tensor in [0, 1] as torch.float32 if numpy == False
    * else numpy.array in [0, 255] as np.uint8

    Args:
        unity_depth (torch.Tensor): one depth map obtained from our simulator
        numpy (bool, optional): Whether to return a float tensor or an int array.
         Defaults to False.
        far: far parameter of the camera in Unity simulator.

    Returns:
        [torch.Tensor or numpy.array]: decoded depth
    """
    R = unity_depth[:, :, 0]
    G = unity_depth[:, :, 1]
    B = unity_depth[:, :, 2]

    R = ((256.0 - R) / 8.0).type(torch.FloatTensor)
    G = ((256.0 - G) / 8.0).type(torch.FloatTensor)
    B = (256.0 - B).type(torch.FloatTensor)
    depth = ((R * 256.0 * 31.0 + G * 256.0 + B).type(torch.FloatTensor)) / (
        256.0 * 31.0 * 31.0 - 1.0
    )
    depth = 1 / depth
    depth = depth.unsqueeze(0)  # (depth * far).unsqueeze(0)

    if log:
        depth = torch.log(depth)
    if normalize:
        depth = depth - torch.min(depth)
        depth /= torch.max(depth)
    if numpy:
        depth = depth.data.cpu().numpy()
        return depth.astype(np.uint8).squeeze()
    return depth


def to_inv_depth(log_depth, numpy=False):
    """Convert log depth tensor to inverse depth image for display

    Args:
        depth (Tensor): log depth float tensor
    """
    depth = torch.exp(log_depth)
    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    inv_depth = 1 / depth
    inv_depth /= torch.max(inv_depth)
    if numpy:
        inv_depth = inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization

    return inv_depth


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
                images_to_save["paths"] += [
                    root / "im_{}_{}_{}.png".format(step, domain, imid)
                ]
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


def get_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def vgg_preprocess(batch):
    """Preprocess batch to use VGG model
    """
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def zero_grad(model: nn.Module):
    """
    Sets gradients to None. Mode efficient than model.zero_grad()
    or opt.zero_grad() according to https://www.youtube.com/watch?v=9mS1fIYj1So

    Args:
        model (nn.Module): model to zero out
    """
    for p in model.parameters():
        p.grad = None
