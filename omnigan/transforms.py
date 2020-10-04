"""Data transforms for the loaders
"""
import torch
import torch.nn.functional as F
from torchvision import transforms as trsfs
import torchvision.transforms.functional as TF
import numpy as np
from scipy.ndimage.interpolation import rotate
from PIL import Image
import traceback
from math import pi


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
        task = tensor = None
        try:
            d = {}
            for task, tensor in data.items():
                d[task] = F.interpolate(
                    tensor, (self.h, self.w), mode=interpolation(task)
                )
            return d
        except Exception as e:
            tb = traceback.format_exc()
            print()
            print(task)
            print(tensor.shape)
            print(interpolation(task))
            print(self.h, self.w)
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


class RandomRotations:
    def __init__(self, p=0.5, angle=5):
        self.p = p
        self.angle = angle

    def cut_balck_edge(self, rotated, angle_selected):
        angle_selected = angle_selected / 180 * pi
        if angle_selected < 15:
            tanAngle = angle_selected
        else:
            tanAngle = np.tan(angle_selected)
        if len(rotated.shape) == 3:
            rotated = rotated.unsqueeze(0)
        _, _, h, w = rotated.shape
        a = int(h * tanAngle / (tanAngle + 1))
        c = int(w * tanAngle / (tanAngle + 1))
        return rotated[:, :, c:-c, a:-a]

    def mapping(self, rotated):
        Dict = {
            0.99607843: 2.0,
            0.9843137: 5.0,
            0.96862745: 9.0,
            0.9764706: 7.0,
            1.0: 1.0,
            0.98039216: 6.0,
            0.99215686: 3.0,
            0.9882353: 4,
            0.9647059: 10,
        }
        tmp = torch.zeros(rotated.shape)
        for k, v in Dict.items():
            tmp += (rotated == k) * v
        return tmp

    def __call__(self, data):
        if self.p > 1 or np.random.rand() > self.p:
            return data
        if isinstance(self.angle, (int, float)):
            selected_angle = self.angle
        elif isinstance(self.angle, (list, tuple)):
            selected_angle = np.random.choice(self.angle)
        else:
            raise NotImplementedError(
                "angle type [%s] is not implemented" % str(type(self.angle))
            )
        if selected_angle == 0:
            return data
        task = tensor = None
        d = {}
        totensor = trsfs.ToTensor()

        for task, tensor in data.items():
            if task == "s":
                self.cut_balck_edge(
                    self.mapping(
                        totensor(
                            TF.rotate(
                                TF.to_pil_image(tensor[0, 0, :, :], "L"),
                                selected_angle,
                                expand=True,
                            )
                        )
                    ),
                    selected_angle,
                )
            else:
                d[task] = torch.tensor(
                    self.cut_balck_edge(
                        rotate(tensor, selected_angle, axes=(3, 2)), selected_angle
                    )
                )
        return d


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
        if opts.data.normalization == "default":
            self.normImage = trsfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif opts.data.normalization == "HRNet":
            self.normImage = trsfs.Normalize(
                ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
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
        return Resize(transform_item.new_size)

    if transform_item.name == "hflip" and not transform_item.ignore:
        return RandomHorizontalFlip(p=transform_item.p or 0.5)
    if transform_item.name == "rot" and not transform_item.ignore:
        return RandomRotations(p=transform_item.p, angle=transform_item.angle)
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
