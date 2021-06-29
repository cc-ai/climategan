"""Data transforms for the loaders
"""
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgba2rgb
from skimage.io import imread
from torchvision import transforms as trsfs
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)

from climategan.tutils import normalize


def interpolation(task):
    if task in ["d", "m", "s"]:
        return {"mode": "nearest"}
    else:
        return {"mode": "bilinear", "align_corners": True}


class Resize:
    def __init__(self, target_size, keep_aspect_ratio=False):
        """
        Resize transform. Target_size can be an int or a tuple of ints,
        depending on whether both height and width should have the same
        final size or not.

        If keep_aspect_ratio is specified then target_size must be an int:
        the smallest dimension of x will be set to target_size and the largest
        dimension will be computed to the closest int keeping the original
        aspect ratio. e.g.
        >>> x = torch.rand(1, 3, 1200, 1800)
        >>> m = torch.rand(1, 1, 600, 600)
        >>> d = {"x": x, "m": m}
        >>> {k: v.shape for k, v in Resize(640, True)(d).items()}
         {"x": (1, 3, 640, 960), "m": (1, 1, 640, 960)}



        Args:
            target_size (int | tuple(int)): New size for the tensor
            keep_aspect_ratio (bool, optional): Whether or not to keep aspect ratio
                when resizing. Requires target_size to be an int. If keeping aspect
                ratio, smallest dim will be set to target_size. Defaults to False.
        """
        if isinstance(target_size, (int, tuple, list)):
            if not isinstance(target_size, int) and not keep_aspect_ratio:
                assert len(target_size) == 2
                self.h, self.w = target_size
            else:
                if keep_aspect_ratio:
                    assert isinstance(target_size, int)
                self.h = self.w = target_size

            self.default_h = int(self.h)
            self.default_w = int(self.w)
            self.sizes = {}
        elif isinstance(target_size, dict):
            assert (
                not keep_aspect_ratio
            ), "dict target_size not compatible with keep_aspect_ratio"

            self.sizes = {
                k: {"h": v, "w": v} for k, v in target_size.items() if k != "default"
            }
            self.default_h = int(target_size["default"])
            self.default_w = int(target_size["default"])

        self.keep_aspect_ratio = keep_aspect_ratio

    def compute_new_default_size(self, tensor):
        """
        compute the new size for a tensor depending on target size
        and keep_aspect_rato

        Args:
            tensor (torch.Tensor): 4D tensor N x C x H x W.

        Returns:
            tuple(int): (new_height, new_width)
        """
        if self.keep_aspect_ratio:
            h, w = tensor.shape[-2:]
            if h < w:
                return (self.h, int(self.default_h * w / h))
            else:
                return (int(self.default_h * h / w), self.default_w)
        return (self.default_h, self.default_w)

    def compute_new_size_for_task(self, task):
        assert (
            not self.keep_aspect_ratio
        ), "compute_new_size_for_task is not compatible with keep aspect ratio"

        if task not in self.sizes:
            return (self.default_h, self.default_w)

        return (self.sizes[task]["h"], self.sizes[task]["w"])

    def __call__(self, data):
        """
        Resize a dict of tensors to the "x" key's new_size

        Args:
            data (dict[str:torch.Tensor]): The data dict to transform

        Returns:
            dict[str: torch.Tensor]: dict with all tensors resized to the
                new size of the data["x"] tensor
        """
        task = tensor = new_size = None
        try:
            if not self.sizes:
                d = {}
                new_size = self.compute_new_default_size(
                    data["x"] if "x" in data else list(data.values())[0]
                )
                for task, tensor in data.items():
                    d[task] = F.interpolate(
                        tensor, size=new_size, **interpolation(task)
                    )
                return d

            d = {}
            for task, tensor in data.items():
                new_size = self.compute_new_size_for_task(task)
                d[task] = F.interpolate(tensor, size=new_size, **interpolation(task))
            return d

        except Exception as e:
            tb = traceback.format_exc()
            print("Debug: task, shape, interpolation, h, w, new_size")
            print(task)
            print(tensor.shape)
            print(interpolation(task))
            print(self.h, self.w)
            print(new_size)
            print(tb)
            raise Exception(e)


class RandomCrop:
    def __init__(self, size, center=False):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)
        self.center = center

    def __call__(self, data):
        H, W = (
            data["x"].size()[-2:] if "x" in data else list(data.values())[0].size()[-2:]
        )

        if not self.center:
            top = np.random.randint(0, H - self.h)
            left = np.random.randint(0, W - self.w)
        else:
            top = (H - self.h) // 2
            left = (W - self.w) // 2

        return {
            task: tensor[:, :, top : top + self.h, left : left + self.w]
            for task, tensor in data.items()
        }


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        # self.flip = TF.hflip
        self.p = p

    def __call__(self, data):
        if np.random.rand() > self.p:
            return data
        return {task: torch.flip(tensor, [3]) for task, tensor in data.items()}


class ToTensor:
    def __init__(self):
        self.ImagetoTensor = trsfs.ToTensor()
        self.MaptoTensor = self.ImagetoTensor

    def __call__(self, data):
        new_data = {}
        for task, im in data.items():
            if task in {"x", "a"}:
                new_data[task] = self.ImagetoTensor(im)
            elif task in {"m"}:
                new_data[task] = self.MaptoTensor(im)
            elif task == "s":
                new_data[task] = torch.squeeze(torch.from_numpy(np.array(im))).to(
                    torch.int64
                )
            elif task == "d":
                new_data = im

        return new_data


class Normalize:
    def __init__(self, opts):
        if opts.data.normalization == "HRNet":
            self.normImage = trsfs.Normalize(
                ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
        else:
            self.normImage = trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normDepth = lambda x: x
        self.normMask = lambda x: x
        self.normSeg = lambda x: x

        self.normalize = {
            "x": self.normImage,
            "s": self.normSeg,
            "d": self.normDepth,
            "m": self.normMask,
        }

    def __call__(self, data):
        return {
            task: self.normalize.get(task, lambda x: x)(tensor.squeeze(0))
            for task, tensor in data.items()
        }


class RandBrightness:  # Input need to be between -1 and 1
    def __call__(self, data):
        return {
            task: rand_brightness(tensor) if task == "x" else tensor
            for task, tensor in data.items()
        }


class RandSaturation:
    def __call__(self, data):
        return {
            task: rand_saturation(tensor) if task == "x" else tensor
            for task, tensor in data.items()
        }


class RandContrast:
    def __call__(self, data):
        return {
            task: rand_contrast(tensor) if task == "x" else tensor
            for task, tensor in data.items()
        }


class BucketizeDepth:
    def __init__(self, opts, domain):
        self.domain = domain

        if opts.gen.d.classify.enable and domain in {"s", "kitti"}:
            self.buckets = torch.linspace(
                *[
                    opts.gen.d.classify.linspace.min,
                    opts.gen.d.classify.linspace.max,
                    opts.gen.d.classify.linspace.buckets - 1,
                ]
            )

            self.transforms = {
                "d": lambda tensor: torch.bucketize(
                    tensor, self.buckets, out_int32=True, right=True
                )
            }
        else:
            self.transforms = {}

    def __call__(self, data):
        return {
            task: self.transforms.get(task, lambda x: x)(tensor)
            for task, tensor in data.items()
        }


class PrepareInference:
    """
    Transform which:
      - transforms a str or an array into a tensor
      - resizes the image to keep the aspect ratio
      - crops in the center of the resized image
      - normalize to 0:1
      - rescale to -1:1
    """

    def __init__(self, target_size=640, half=False, is_label=False, enforce_128=True):
        if enforce_128:
            if target_size % 2 ** 7 != 0:
                raise ValueError(
                    f"Received a target_size of {target_size}, which is not a "
                    + "multiple of 2^7 = 128. Set enforce_128 to False to disable "
                    + "this error."
                )
        self.resize = Resize(target_size, keep_aspect_ratio=True)
        self.crop = RandomCrop((target_size, target_size), center=True)
        self.half = half
        self.is_label = is_label

    def process(self, t):
        if isinstance(t, (str, Path)):
            t = imread(str(t))

        if isinstance(t, np.ndarray):
            if t.shape[-1] == 4:
                t = rgba2rgb(t)

            t = torch.from_numpy(t)
            if t.ndim == 3:
                t = t.permute(2, 0, 1)

        if t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)

        if not self.is_label:
            t = t.to(torch.float32)
            t = normalize(t)
            t = (t - 0.5) * 2

        t = {"m": t} if self.is_label else {"x": t}
        t = self.resize(t)
        t = self.crop(t)
        t = t["m"] if self.is_label else t["x"]

        if self.half and not self.is_label:
            t = t.half()

        return t

    def __call__(self, x):
        """
        normalize, rescale, resize, crop in the center

        x can be: dict {"task": data} list [data, ..] or data
        data ^ can be a str, a Path, a numpy arrray or a Tensor
        """
        if isinstance(x, dict):
            return {k: self.process(v) for k, v in x.items()}

        if isinstance(x, list):
            return [self.process(t) for t in x]

        return self.process(x)


class PrepareTest:
    """
    Transform which:
      - transforms a str or an array into a tensor
      - resizes the image to keep the aspect ratio
      - crops in the center of the resized image
      - normalize to 0:1 (optional)
      - rescale to -1:1 (optional)
    """

    def __init__(self, target_size=640, half=False):
        self.resize = Resize(target_size, keep_aspect_ratio=True)
        self.crop = RandomCrop((target_size, target_size), center=True)
        self.half = half

    def process(self, t, normalize=False, rescale=False):
        if isinstance(t, (str, Path)):
            # t = img_as_float(imread(str(t)))
            t = imread(str(t))
            if t.shape[-1] == 4:
                # t = rgba2rgb(t)
                t = t[:, :, :3]
            if np.ndim(t) == 2:
                t = np.repeat(t[:, :, np.newaxis], 3, axis=2)

        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
            t = t.permute(2, 0, 1)

        if len(t.shape) == 3:
            t = t.unsqueeze(0)

        t = t.to(torch.float32)
        normalize(t) if normalize else t
        (t - 0.5) * 2 if rescale else t
        t = {"x": t}
        t = self.resize(t)
        t = self.crop(t)
        t = t["x"]

        if self.half:
            return t.to(torch.float16)

        return t

    def __call__(self, x, normalize=False, rescale=False):
        """
        Call process()

        x can be: dict {"task": data} list [data, ..] or data
        data ^ can be a str, a Path, a numpy arrray or a Tensor
        """
        if isinstance(x, dict):
            return {k: self.process(v, normalize, rescale) for k, v in x.items()}

        if isinstance(x, list):
            return [self.process(t, normalize, rescale) for t in x]

        return self.process(x, normalize, rescale)


def get_transform(transform_item, mode):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomCrop(
            (transform_item.height, transform_item.width),
            center=transform_item.center == mode,
        )

    elif transform_item.name == "resize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return Resize(
            transform_item.new_size, transform_item.get("keep_aspect_ratio", False)
        )

    elif transform_item.name == "hflip" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "brightness" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandBrightness()

    elif transform_item.name == "saturation" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandSaturation()

    elif transform_item.name == "contrast" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandContrast()

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode, domain):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    transforms = []
    color_jittering_transforms = ["brightness", "saturation", "contrast"]

    for t in opts.data.transforms:
        if t.name not in color_jittering_transforms:
            transforms.append(get_transform(t, mode))

    if "p" not in opts.tasks and mode == "train":
        for t in opts.data.transforms:
            if t.name in color_jittering_transforms:
                transforms.append(get_transform(t, mode))

    transforms += [Normalize(opts), BucketizeDepth(opts, domain)]
    transforms = [t for t in transforms if t is not None]

    return transforms


# ----- Adapted functions from https://github.com/mit-han-lab/data-efficient-gans -----#
def rand_brightness(tensor, is_diff_augment=False):
    if is_diff_augment:
        assert len(tensor.shape) == 4
        type_ = tensor.dtype
        device_ = tensor.device
        rand_tens = torch.rand(tensor.size(0), 1, 1, 1, dtype=type_, device=device_)
        return tensor + (rand_tens - 0.5)
    else:
        factor = random.uniform(0.5, 1.5)
        tensor = adjust_brightness(tensor, brightness_factor=factor)
        # dummy pixels to fool scaling and preserve range
        tensor[:, :, 0, 0] = 1.0
        tensor[:, :, -1, -1] = 0.0
        return tensor


def rand_saturation(tensor, is_diff_augment=False):
    if is_diff_augment:
        assert len(tensor.shape) == 4
        type_ = tensor.dtype
        device_ = tensor.device
        rand_tens = torch.rand(tensor.size(0), 1, 1, 1, dtype=type_, device=device_)
        x_mean = tensor.mean(dim=1, keepdim=True)
        return (tensor - x_mean) * (rand_tens * 2) + x_mean
    else:
        factor = random.uniform(0.5, 1.5)
        tensor = adjust_saturation(tensor, saturation_factor=factor)
        # dummy pixels to fool scaling and preserve range
        tensor[:, :, 0, 0] = 1.0
        tensor[:, :, -1, -1] = 0.0
        return tensor


def rand_contrast(tensor, is_diff_augment=False):
    if is_diff_augment:
        assert len(tensor.shape) == 4
        type_ = tensor.dtype
        device_ = tensor.device
        rand_tens = torch.rand(tensor.size(0), 1, 1, 1, dtype=type_, device=device_)
        x_mean = tensor.mean(dim=[1, 2, 3], keepdim=True)
        return (tensor - x_mean) * (rand_tens + 0.5) + x_mean
    else:
        factor = random.uniform(0.5, 1.5)
        tensor = adjust_contrast(tensor, contrast_factor=factor)
        # dummy pixels to fool scaling and preserve range
        tensor[:, :, 0, 0] = 1.0
        tensor[:, :, -1, -1] = 0.0
        return tensor


def rand_cutout(tensor, ratio=0.5):
    assert len(tensor.shape) == 4, "For rand cutout, tensor must be 4D."
    type_ = tensor.dtype
    device_ = tensor.device
    cutout_size = int(tensor.size(-2) * ratio + 0.5), int(tensor.size(-1) * ratio + 0.5)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(tensor.size(0), dtype=torch.long, device=device_),
        torch.arange(cutout_size[0], dtype=torch.long, device=device_),
        torch.arange(cutout_size[1], dtype=torch.long, device=device_),
    )
    size_ = [tensor.size(0), 1, 1]
    offset_x = torch.randint(
        0,
        tensor.size(-2) + (1 - cutout_size[0] % 2),
        size=size_,
        device=device_,
    )
    offset_y = torch.randint(
        0,
        tensor.size(-1) + (1 - cutout_size[1] % 2),
        size=size_,
        device=device_,
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=tensor.size(-2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=tensor.size(-1) - 1
    )
    mask = torch.ones(
        tensor.size(0), tensor.size(2), tensor.size(3), dtype=type_, device=device_
    )
    mask[grid_batch, grid_x, grid_y] = 0
    return tensor * mask.unsqueeze(1)


def rand_translation(tensor, ratio=0.125):
    assert len(tensor.shape) == 4, "For rand translation, tensor must be 4D."
    device_ = tensor.device
    shift_x, shift_y = (
        int(tensor.size(2) * ratio + 0.5),
        int(tensor.size(3) * ratio + 0.5),
    )
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[tensor.size(0), 1, 1], device=device_
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[tensor.size(0), 1, 1], device=device_
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(tensor.size(0), dtype=torch.long, device=device_),
        torch.arange(tensor.size(2), dtype=torch.long, device=device_),
        torch.arange(tensor.size(3), dtype=torch.long, device=device_),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, tensor.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, tensor.size(3) + 1)
    x_pad = F.pad(tensor, [1, 1, 1, 1, 0, 0, 0, 0])
    tensor = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return tensor


class DiffTransforms:
    def __init__(self, diff_aug_opts):
        self.do_color_jittering = diff_aug_opts.do_color_jittering
        self.do_cutout = diff_aug_opts.do_cutout
        self.do_translation = diff_aug_opts.do_translation
        self.cutout_ratio = diff_aug_opts.cutout_ratio
        self.translation_ratio = diff_aug_opts.translation_ratio

    def __call__(self, tensor):
        if self.do_color_jittering:
            tensor = rand_brightness(tensor, is_diff_augment=True)
            tensor = rand_contrast(tensor, is_diff_augment=True)
            tensor = rand_saturation(tensor, is_diff_augment=True)
        if self.do_translation:
            tensor = rand_translation(tensor, ratio=self.translation_ratio)
        if self.do_cutout:
            tensor = rand_cutout(tensor, ratio=self.cutout_ratio)
        return tensor
