"""Data transforms for the loaders
"""
import torch
import torch.nn.functional as F
from torchvision import transforms as trsfs
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


def interpolation(task):
    if task in ["d", "m", "s"]:
        return "nearest"
    else:
        return "bilinear"


class Resize:
    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if not isinstance(target_size, int):
            assert len(target_size) == 2
            self.h, self.w = target_size
        else:
            self.h = self.w = target_size

        self.h = int(self.h)
        self.w = int(self.w)

    def __call__(self, data):
        return {
            task: F.interpolate(tensor, (self.h, self.w), mode=interpolation(task))
            for task, tensor in data.items()
        }


class RandomCrop:
    def __init__(self, size):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            self.h, self.w = size
        else:
            assert len(size == 2)
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)

    def __call__(self, data):
        h, w = data["x"].size[-2:]
        top = np.random.randint(0, h - self.h)
        left = np.random.randint(0, w - self.w)
        return {
            task: tensor[:, top : top + self.h, left + self.w]
            for task, tensor in data.items()
        }


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        # self.flip = TF.hflip
        self.p = p

    def __call__(self, data):
        if np.random.rand() > self.p:
            return data
        for task, tensor in data.items():
            print(task, tensor.shape)
        return {task: torch.flip(tensor, (1,)) for task, tensor in data.items()}


class ToTensor:
    def __init__(self):
        self.ImagetoTensor = trsfs.ToTensor()
        self.MaptoTensor = self.ImagetoTensor

    def __call__(self, data):
        new_data = {}
        for task, im in data.items():
            if task in {"x", "a", "d"}:
                new_data[task] = self.ImagetoTensor(im)
            elif task in {"h", "w", "m"}:
                new_data[task] = self.MaptoTensor(im)
            elif task == "s":
                new_data[task] = torch.squeeze(torch.from_numpy(np.array(im))).to(
                    torch.int64
                )

        return new_data


class Normalize:
    def __init__(self):
        # self.normImage = trsfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normImage = trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normSeg = lambda x: x
        # self.normSeg = trsfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.normDepth = lambda x: x  # trsfs.Normalize([1 / 255], [1 / 3])
        self.normMask = lambda x: x

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


def get_transform(transform_item):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not transform_item.ignore:
        return RandomCrop((transform_item.height, transform_item.width))

    if transform_item.name == "resize" and not transform_item.ignore:
        return Resize(transform_item.new_size)

    if transform_item.name == "hflip" and not transform_item.ignore:
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    if transform_item.ignore:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item)
    """
    last_transforms = [Normalize()]

    conf_transforms = []
    for t in opts.data.transforms:
        if get_transform(t) is not None:
            conf_transforms.append(get_transform(t))

    return conf_transforms + last_transforms
