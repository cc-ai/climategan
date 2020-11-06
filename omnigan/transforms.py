"""Data transforms for the loaders
"""
import torch
import torch.nn.functional as F
from torchvision import transforms as trsfs
import numpy as np
from PIL import Image
import traceback


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
        assert isinstance(target_size, (int, tuple, list))
        if not isinstance(target_size, int) and not keep_aspect_ratio:
            assert len(target_size) == 2
            self.h, self.w = target_size
        else:
            if keep_aspect_ratio:
                assert isinstance(target_size, int)
            self.h = self.w = target_size

        self.keep_aspect_ratio = keep_aspect_ratio

        self.h = int(self.h)
        self.w = int(self.w)

    def compute_new_size(self, tensor):
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
                return (self.h, int(self.h * w / h))
            else:
                return (int(self.h * h / w), self.w)
        return (self.h, self.w)

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
            d = {}
            new_size = self.compute_new_size(data["x"])
            for task, tensor in data.items():
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
        h, w = data["x"].size()[-2:]
        top = np.random.randint(0, h - self.h)
        left = np.random.randint(0, w - self.w)
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
        self.normDepth = lambda x: x  # trsfs.Normalize([1 / 255], [1 / 3])
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


def get_transform(transform_item):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not transform_item.ignore:
        return RandomCrop((transform_item.height, transform_item.width))

    if transform_item.name == "resize" and not transform_item.ignore:
        return Resize(
            transform_item.new_size, transform_item.get("keep_aspect_ratio", False)
        )

    if transform_item.name == "hflip" and not transform_item.ignore:
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    if transform_item.ignore:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item)
    """
    last_transforms = [Normalize(opts)]

    conf_transforms = []
    for t in opts.data.transforms:
        if get_transform(t) is not None:
            conf_transforms.append(get_transform(t))

    return conf_transforms + last_transforms
