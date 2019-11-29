import os
from copy import copy
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trsf
from numpy.random import permutation

from .utils import env_to_path

# -----------------------------
# -----  Image utilities  -----
# -----------------------------

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


def get_images_from_dir(data_dir):
    """Return a list of paths of images within specified data_dir
    """
    images = []
    assert os.path.isdir(data_dir), "%s is not a valid directory" % data_dir

    for root, _, fnames in sorted(os.walk(data_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def get_images_from_file(file_path):
    """Return a list of paths of images listed in file_path (1 per line)
    """
    with open(file_path, "r") as f:
        imgs = [l.strip() for l in f.readlines()]
    return [i for i in imgs if is_image_file(i)]


def default_reader(path):
    """get an Image opject from a path, converted in rgb
    """
    return Image.open(path).convert("RGB")


def is_image_file(filename):
    """Check that a file's name points to a known image format
    """
    return Path(filename).suffix in IMG_EXTENSIONS


# ---------------------------------
# -----  Transform utilities  -----
# ---------------------------------


def get_transform(transform_item):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.loaders.transforms
    """
    if transform_item.name == "crop" and not transform_item.ignore:
        return trsf.RandomCrop((transform_item.height, transform_item.width))

    if transform_item.name == "resize" and not transform_item.ignore:
        return trsf.Resize(transform_item.new_size)

    if transform_item.name == "hflip" and not transform_item.ignore:
        return trsf.RandomHorizontalFlip(p=transform_item.p or 0.5)

    raise ValueError("Unknown transform_item")


def get_all_transforms(transform_items):
    """Get all the transform functions listed in opts.data.loaders.transforms
    using get_transform(transform_item)
    """
    last_transforms = [
        trsf.ToTensor(),
        trsf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    conf_transforms = []
    for transform_item in transform_items:
        conf_transforms.append(get_transform(transform_item))
    transform_list = conf_transforms + last_transforms
    return trsf.Compose(transform_list)


# -------------------------------
# -----  Loaders utilities  -----
# -------------------------------


def get_all_loaders(opts):
    B_conf = copy(opts.loaders)
    A_conf = copy(opts.loaders)
    B_val_loader = A_val_loader = None

    if opts.dirs.base:
        B_conf.data_dir = env_to_path(str(Path(opts.dirs.base) / opts.dirs.train.B))
        A_conf.data_dir = env_to_path(str(Path(opts.dirs.base) / opts.dirs.train.A))
    if opts.files.base:
        B_conf.data_file = str(Path(opts.files.base) / opts.files.train.B)
        A_conf.data_file = env_to_path(str(Path(opts.files.base) / opts.files.train.A))

    A_loader = get_loader(**A_conf)
    B_loader = get_loader(**B_conf)

    if (opts.dirs.val and opts.dirs.val.A) or (opts.files.val and opts.files.val.A):
        A_val_conf = copy(opts.loaders)
        if opts.dirs.base:
            A_val_conf.data_dir = env_to_path(
                str(Path(opts.dirs.base) / opts.dirs.val.A)
            )
        if opts.files.base:
            A_val_conf.data_file = env_to_path(
                str(Path(opts.files.base) / opts.files.val.A)
            )
        A_val_loader = get_loader(**A_val_conf)

    if (opts.dirs.val and opts.dirs.val.B) or (opts.files.val and opts.files.val.B):
        B_val_conf = copy(opts.loaders)
        if opts.dirs.base:
            B_val_conf.data_dir = env_to_path(
                str(Path(opts.dirs.base) / opts.dirs.val.B)
            )
        if opts.files.base:
            B_val_conf.data_file = env_to_path(
                str(Path(opts.files.base) / opts.files.val.B)
            )
        B_val_loader = get_loader(**B_val_conf)

    return A_loader, B_loader, A_val_loader, B_val_loader


def get_loader(
    data_dir=None,
    data_file=None,
    batch_size=2,
    transforms=None,
    shuffle=False,
    num_workers=None,
    return_paths=False,
    max_load_size=-1,
    shuffle_before_max_load_size=True,
    **kwargs
):

    if (data_dir is not None) or (data_file is not None):
        raise ValueError("Both data_dir and data_file are None")

    return DataLoader(
        OmniDataset(
            data_dir,
            data_file,
            return_paths,
            transform=get_all_transforms(transforms),
            max_load_size=max_load_size,
            shuffle_before_max_load_size=shuffle_before_max_load_size,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ---------------------
# -----  Dataset  -----
# ---------------------


class OmniDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        data_file=None,
        return_paths=False,
        transform=None,
        max_load_size=-1,
        shuffle_before_max_load_size=True,
    ):
        """OmniGAN Dataset

        Args:
            data_dir (str, optional): Path to directory of images.
                Has priority over data_file. Defaults to None.
            data_file (str, optional): Path to file listing the dataset's images.
                Defaults to None.
            return_paths (bool, optional): Do you want to just return images
                or also their path (img, path). Defaults to False.
            transform (torchvision.Transform, optional): The transformation(s) to apply
                to images. Defaults to None.
            max_load_size (int, optional): Whether to load a limited amount of images
                or all of them (-1). Defaults to -1.
            shuffle_before_max_load_size (bool, optional): If limiting the load_size,
                do you want to get the first n one or n random ones? Defaults to True.

        Returns:
            [type]: [description]
        """
        super().__init__()
        self.data_dir = data_dir
        self.imgs = (
            get_images_from_dir(data_dir)
            if data_file is None
            else get_images_from_file(data_file)
        )
        if max_load_size and max_load_size > 0:
            if shuffle_before_max_load_size:
                self.imgs = [
                    self.imgs[i] for i in permutation(len(self.imgs))[:max_load_size]
                ]
            else:
                self.imgs = self.imgs[:max_load_size]
        self.transform = transform
        self.return_paths = return_paths
        self.reader = default_reader

    def __getitem__(self, index):
        path = self.imgs[index]

        img = self.reader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        return img

    def __len__(self):
        return len(self.imgs)
