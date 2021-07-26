"""Tensor-utils
"""
import io
import math
from contextlib import redirect_stdout
from pathlib import Path

# from copy import copy
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from skimage import io as skio
from torch import autograd
from torch.autograd import Variable
from torch.nn import init

from climategan.utils import all_texts_to_array


def transforms_string(ts):
    return " -> ".join([t.__class__.__name__ for t in ts.transforms])


def init_weights(net, init_type="normal", init_gain=0.02, verbose=0, caller=""):
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
        print(
            "init_weights({}): init_type is {}, defaulting to normal".format(
                caller + " " + net.__class__.__name__, init_type
            )
        )
        init_type = "normal"
    if not init_gain:
        print(
            "init_weights({}): init_gain is {}, defaulting to normal".format(
                caller + " " + net.__class__.__name__, init_type
            )
        )
        init_gain = 0.02

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
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
        image = tensor
        if image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)

    if image.min() < 0 and image.min() > -1:
        image = image / 2 + 0.5
    elif image.min() < -1:
        raise ValueError("can't handle this data")

    skimage.io.imshow(image)


def normalize_tensor(t):
    """
    Brings any tensor to the [0; 1] range.

    Args:
        t (torch.Tensor): input to normalize

    Returns:
        torch.Tensor: t projected to [0; 1]
    """
    t = t - torch.min(t)
    t = t / torch.max(t)
    return t


def get_normalized_depth_t(tensor, domain, normalize=False, log=True):
    assert not (normalize and log)
    if domain == "r":
        # megadepth depth
        tensor = tensor.unsqueeze(0)
        tensor = tensor - torch.min(tensor)
        tensor = torch.true_divide(tensor, torch.max(tensor))

    elif domain == "s":
        # from 3-channel depth encoding from Unity simulator to 1-channel [0-1] values
        tensor = decode_unity_depth_t(tensor, log=log, normalize=normalize)

    elif domain == "kitti":
        tensor = tensor / 100
        if not log:
            tensor = 1 / tensor
            if normalize:
                tensor = tensor - tensor.min()
                tensor = tensor / tensor.max()
        else:
            tensor = torch.log(tensor)

        tensor = tensor.unsqueeze(0)

    return tensor


def decode_bucketed_depth(tensor, opts):
    # tensor is size 1 x C x H x W
    assert tensor.shape[0] == 1
    idx = torch.argmax(tensor.squeeze(0), dim=0)  # channels become dim 0 with squeeze
    linspace_args = (
        opts.gen.d.classify.linspace.min,
        opts.gen.d.classify.linspace.max,
        opts.gen.d.classify.linspace.buckets,
    )
    indexer = torch.linspace(*linspace_args)
    log_depth = indexer[idx.long()].to(torch.float32)  # H x W
    depth = torch.exp(log_depth)
    return depth.unsqueeze(0).unsqueeze(0).to(tensor.device)


def decode_unity_depth_t(unity_depth, log=True, normalize=False, numpy=False, far=1000):
    """Transforms the 3-channel encoded depth map from our Unity simulator
    to 1-channel depth map containing metric depth values.
    The depth is encoded in the following way:
    - The information from the simulator is (1 - LinearDepth (in [0,1])).
        far corresponds to the furthest distance to the camera included in the
        depth map.
        LinearDepth * far gives the real metric distance to the camera.
    - depth is first divided in 31 slices encoded in R channel with values ranging
        from 0 to 247
    - each slice is divided again in 31 slices, whose value is encoded in G channel
    - each of the G slices is divided into 256 slices, encoded in B channel

    In total, we have a discretization of depth into N = 31*31*256 - 1 possible values,
    covering a range of far/N meters.

    Note that, what we encode here is 1 - LinearDepth so that the furthest point is
    [0,0,0] (that is sky) and the closest point[255,255,255]

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

    R = ((247 - R) / 8).type(torch.IntTensor)
    G = ((247 - G) / 8).type(torch.IntTensor)
    B = (255 - B).type(torch.IntTensor)
    depth = ((R * 256 * 31 + G * 256 + B).type(torch.FloatTensor)) / (256 * 31 * 31 - 1)
    depth = depth * far
    if not log:
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
    # visualize prediction using inverse depth, so that we don't need sky
    # segmentation (if you want to use RGB map for visualization,
    # you have to run semantic segmentation to mask the sky first
    # since the depth of sky is random from CNN)
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
    """Preprocess batch to use VGG model"""
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


# Take the prediction of fake and real images from the combined batch
def divide_pred(disc_output):
    """
    Divide a multiscale discriminator's output into 2 sets of tensors,
    expecting the input to the discriminator to be a concatenation
    on the batch axis of real and fake (or fake and real) images,
    effectively doubling the batch size for better batchnorm statistics

    Args:
        disc_output (list | torch.Tensor): Discriminator output to split

    Returns:
        list | torch.Tensor[type]: pair of split outputs
    """
    # https://github.com/NVlabs/SPADE/blob/master/models/pix2pix_model.py
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(disc_output) == list:
        half1 = []
        half2 = []
        for p in disc_output:
            half1.append([tensor[: tensor.size(0) // 2] for tensor in p])
            half2.append([tensor[tensor.size(0) // 2 :] for tensor in p])
    else:
        half1 = disc_output[: disc_output.size(0) // 2]
        half2 = disc_output[disc_output.size(0) // 2 :]

    return half1, half2


def is_tpu_available():
    _torch_tpu_available = False
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        if "xla" in str(xm.xla_device()):
            _torch_tpu_available = True
        else:
            _torch_tpu_available = False
    except ImportError:
        _torch_tpu_available = False

    return _torch_tpu_available


def get_WGAN_gradient(input, output):
    # github code reference:
    # https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    # Calculate the gradient that WGAN-gp needs
    grads = autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones(output.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def print_num_parameters(trainer, force=False):
    if trainer.verbose == 0 and not force:
        return
    print("-" * 35)
    if trainer.G.encoder is not None:
        print(
            "{:21}:".format("num params encoder"),
            f"{get_num_params(trainer.G.encoder):12,}",
        )
    for d in trainer.G.decoders.keys():
        print(
            "{:21}:".format(f"num params decoder {d}"),
            f"{get_num_params(trainer.G.decoders[d]):12,}",
        )

    print(
        "{:21}:".format("num params painter"),
        f"{get_num_params(trainer.G.painter):12,}",
    )

    if trainer.D is not None:
        for d in trainer.D.keys():
            print(
                "{:21}:".format(f"num params discrim {d}"),
                f"{get_num_params(trainer.D[d]):12,}",
            )

    print("-" * 35)


def srgb2lrgb(x):
    x = normalize(x)
    im = ((x + 0.055) / 1.055) ** (2.4)
    im[x <= 0.04045] = x[x <= 0.04045] / 12.92
    return im


def lrgb2srgb(ims):
    if len(ims.shape) == 3:
        ims = [ims]
        stack = False
    else:
        ims = list(ims)
        stack = True

    outs = []
    for im in ims:

        out = torch.zeros_like(im)
        for k in range(3):
            temp = im[k, :, :]

            out[k, :, :] = 12.92 * temp * (temp <= 0.0031308) + (
                1.055 * torch.pow(temp, (1 / 2.4)) - 0.055
            ) * (temp > 0.0031308)
        outs.append(out)

    if stack:
        return torch.stack(outs)

    return outs[0]


def normalize(t, mini=0, maxi=1):
    if len(t.shape) == 3:
        return mini + (maxi - mini) * (t - t.min()) / (t.max() - t.min())

    batch_size = t.shape[0]
    min_t = t.reshape(batch_size, -1).min(1)[0].reshape(batch_size, 1, 1, 1)
    t = t - min_t
    max_t = t.reshape(batch_size, -1).max(1)[0].reshape(batch_size, 1, 1, 1)
    t = t / max_t
    return mini + (maxi - mini) * t


def retrieve_sky_mask(seg):
    """
    get the binary mask for the sky given a segmentation tensor
    of logits (N x C x H x W) or labels (N x H x W)

    Args:
        seg (torch.Tensor): Segmentation map

    Returns:
        torch.Tensor: Sky mask
    """
    if len(seg.shape) == 4:  # Predictions
        seg_ind = torch.argmax(seg, dim=1)
    else:
        seg_ind = seg

    sky_mask = seg_ind == 9
    return sky_mask


def all_texts_to_tensors(texts, width=640, height=40):
    """
    Creates a list of tensors with texts from PIL images

    Args:
        texts (list(str)): texts to write
        width (int, optional): width of individual texts. Defaults to 640.
        height (int, optional): height of individual texts. Defaults to 40.

    Returns:
        list(torch.Tensor): len(texts) tensors 3 x height x width
    """
    arrays = all_texts_to_array(texts, width, height)
    arrays = [array.transpose(2, 0, 1) for array in arrays]
    return [torch.tensor(array) for array in arrays]


def write_architecture(trainer):
    stem = "archi"
    out = Path(trainer.opts.output_path)

    # encoder
    with open(out / f"{stem}_encoder.txt", "w") as f:
        f.write(str(trainer.G.encoder))

    # decoders
    for k, v in trainer.G.decoders.items():
        with open(out / f"{stem}_decoder_{k}.txt", "w") as f:
            f.write(str(v))

    # painter
    if get_num_params(trainer.G.painter) > 0:
        with open(out / f"{stem}_painter.txt", "w") as f:
            f.write(str(trainer.G.painter))

    # discriminators
    if get_num_params(trainer.D) > 0:
        for k, v in trainer.D.items():
            with open(out / f"{stem}_discriminator_{k}.txt", "w") as f:
                f.write(str(v))

    with io.StringIO() as buf, redirect_stdout(buf):
        print_num_parameters(trainer)
        output = buf.getvalue()
        with open(out / "archi_num_params.txt", "w") as f:
            f.write(output)


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (  # noqa: E731
        torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )


def mix_noise(x, mask, res=(8, 3), weight=0.1):
    noise = rand_perlin_2d(x.shape[-2:], res).unsqueeze(0).unsqueeze(0).to(x.device)
    noise = noise - noise.min()
    mask = mask.repeat(1, 3, 1, 1).to(x.device).to(torch.float16)
    y = mask * (weight * noise + (1 - weight) * x) + (1 - mask) * x
    return y


def tensor_ims_to_np_uint8s(ims):
    """
    transform a CHW of NCHW tensor into a list of np.uint8 [0, 255]
    image arrays

    Args:
        ims (torch.Tensor | list): [description]
    """
    if not isinstance(ims, list):
        assert isinstance(ims, torch.Tensor)
        if ims.ndim == 3:
            ims = [ims]

    nps = []
    for t in ims:
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        else:
            assert t.shape[-1] == 3

        n = t.cpu().numpy()
        n = (n + 1) / 2 * 255
        nps.append(n.astype(np.uint8))

    return nps[0] if len(nps) == 1 else nps
