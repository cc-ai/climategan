from torchvision import transforms as trsfs
import numpy as np
from addict import Dict


class Resize:
    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if not isinstance(target_size, int):
            self.h, self.w = target_size
        else:
            assert len(target_size == 2)
            self.h = self.w = target_size

        self.h = int(self.h)
        self.w = int(self.w)

    def __call__(self, sample):
        return Dict(
            {
                task: sktransform.resize(array, (self.h, self.w))
                for task, array in sample.items()
            }
        )


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

    def __call__(self, sample):
        h, w = sample.x.shape[-2:]
        top = np.random.randint(0, h - self.h)
        left = np.random.randint(0, w - self.w)

        return Dict(
            {k: v[top : top + self.h, left : left + self.w] for k, v in sample.items()}
        )


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.flip = trsfs.RandomVerticalFlip(1)
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        return Dict({task: self.flip(array) for task, array in sample.items()})


class ToTensor:
    def __init__(self):
        self.ImagetoTensor = trsfs.ToTensor()
        self.MaptoTensor = self.ImagetoTensor

    def __call__(self, sample):
        new_sample = Dict()
        for task, array in sample.items():
            if task in {"x", "s"}:
                new_sample[task] = self.ImagetoTensor(array)
            elif task in {"h", "d"}:
                new_sample[task] = self.MaptoTensor(array)
        return new_sample


class Normalize:
    def __init__(self):
        self.normaImage = trsfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normSeg = trsfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.normDepth = trsfs.Normalize([1 / 255], [1 / 3])

        self.normalize = {
            "x": self.normaImage,
            "s": self.normSeg,
            "d": self.normDepth,
        }

    def __call__(self, sample):
        return Dict(
            {
                task: self.normalize[task](array) if task in self.normalize else array
                for task, array in sample.items()
            }
        )


def get_transform(transform_item):
    """Returns the torchivion transform function associated to a
    transform_item listed in conf.data.transforms ; transform_item is
    an addict.Dict
    """
    if transform_item.name == "crop" and not transform_item.ignore:
        return RandomCrop((transform_item.height, transform_item.width))

    if transform_item.name == "resize" and not transform_item.ignore:
        return Resize(transform_item.new_size)

    if transform_item.name == "hflip" and not transform_item.ignore:
        return RandomVerticalFlip(p=transform_item.p or 0.5)

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts):
    """Get all the transform functions listed in conf.data.transforms
    using get_transform(transform_item)
    """
    last_transforms = [
        ToTensor(),
        Normalize(),
    ]

    for t in opts.data.transforms:
        conf_transforms = []
        conf_transforms.append(get_transform(t))

    return conf_transforms + last_transforms
