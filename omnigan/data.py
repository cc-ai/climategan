"""Data-loading functions in order to create a Dataset and DataLoaders.
Transforms for loaders are in transforms.py
"""

from pathlib import Path
import yaml
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trsfs
from imageio import imread
from torchvision import transforms
import numpy as np
from .transforms import get_transforms
from .transforms import ToTensor
from PIL import Image
from omnigan.tutils import get_normalized_depth_t

# ? paired dataset

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)

classes_dict = {
    "s": {
        (0, 0, 255, 255): 0,  # Water
        (55, 55, 55, 255): 1,  # Ground
        (0, 255, 255, 255): 2,  # Building
        (255, 212, 0, 255): 3,  # Traffic items
        (0, 255, 0, 255): 4,  # Vegetation
        (255, 97, 0, 255): 5,  # Terrain
        (255, 0, 0, 255): 6,  # Car
        (0, 0, 0, 0): 7,  # Trees
        (255, 0, 255, 255): 8,  # Person
        (0, 0, 0, 255): 9,  # Sky
        (255, 255, 255, 255): 10,  # Default
    },
    "r": {
        (0, 0, 255, 255): 0,  # Water NOT FOUND YET
        (55, 55, 55, 255): 1,  # Ground
        (0, 255, 255, 255): 2,  # Building
        (255, 212, 0, 255): 3,  # Traffic items
        # (0, 255, 0, 255): 4,  # Vegetation NOT FOUND YET
        (255, 97, 0, 255): 5,  # Terrain
        (255, 0, 0, 255): 6,  # Car
        (0, 255, 0, 255): 7,  # Trees
        (220, 20, 60, 255): 8,  # Person
        (8, 19, 49, 255): 9,  # Sky
        (0, 80, 100, 255): 10,  # Default, object with no class?
    },
}


def decode_segmap_unity_labels(tensor, nc=11):
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Arguments:
        image {array} -- segmented image
        (array of image size containing class at each pixel)
    Returns:
        array of size 3*nc -- A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((nc, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap = torch.tensor(colormap)

    r = torch.zeros((tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8)
    g = torch.zeros((tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8)
    b = torch.zeros((tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8)
    for l in range(nc):
        idx = torch.argmax(tensor.squeeze(0), dim=0)
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = torch.stack([r, g, b], dim=0)
    return rgb.unsqueeze(0)


def decode_segmap_cityscapes_labels(image, nc=19):
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Arguments:
        image {array} -- segmented image
        (array of image size containing class at each pixel)
    Returns:
        array of size 3*nc -- A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((19, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(nc):
        idx = image == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def find_closest_class(pixel, domain):
    """Takes a pixel as input and finds the closest known pixel value corresponding to a class
    Arguments:
        pixel -- tuple pixel (R,G,B,A)
        domain -- domain of the image ("r" or "s")
    Returns:
        tuple pixel (R,G,B,A) corresponding to a key in classes_dict
    """
    min_dist = float("inf")
    closest_pixel = None
    for pixel_value in classes_dict[domain].keys():
        dist = np.sum(np.absolute(np.subtract(pixel, pixel_value)))
        if dist < min_dist:
            min_dist = dist
            closest_pixel = pixel_value
    return closest_pixel


def encode_segmap(arr, domain, num_classes=11):
    """Change a segmentation RGBA array to a segmentation array with each channel being a binary array of one class
    Arguments:
        numpy array -- segmented image (H) x (W) x (4 RGBA values)
    Returns:
        numpy array of size (num_classes) x (H) x (W)
    """
    new_arr = np.zeros((num_classes, arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            pixel_rgba = tuple(arr[i, j, :])
            if pixel_rgba in classes_dict[domain].keys():
                new_arr[classes_dict[domain][pixel_rgba], i, j] = 1.0
            else:
                pixel_rgba_closest = find_closest_class(pixel_rgba, domain)
                new_arr[classes_dict[domain][pixel_rgba_closest], i, j] = 1.0
    return new_arr


def is_image_file(filename):
    """Check that a file's name points to a known image format
    """
    return Path(filename).suffix in IMG_EXTENSIONS


def pil_image_loader(path, task, domain):
    if task == "s" and domain == "s":
        arr = np.array(Image.open(path).convert("RGBA"))
        arr = encode_segmap(arr, domain)
        return arr
    elif Path(path).suffix == ".npy":
        arr = np.load(path).astype(np.uint8)
    elif is_image_file(path):
        # arr = imread(path).astype(np.uint8)
        arr = np.array(Image.open(path).convert("RGB"))
    else:
        raise ValueError("Unknown data type {}".format(path))

    # Convert from RGBA to RGB for images
    if len(arr.shape) == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, 0:3]

    if task == "m":
        arr[arr != 0] = 255
        # Make sure mask is single-channel
        if len(arr.shape) >= 3:
            arr = arr[:, :, 0]

    # assert len(arr.shape) == 3, (path, task, arr.shape)

    return Image.fromarray(arr)


def tensor_loader(path, task, domain):
    """load data as tensors
    Args:
        path (str): path to data
        task (str):
        domain 
    Returns:
        [Tensor]: C x H x W
    """
    if task == "s" and domain == "s":
        arr = np.array(Image.open(path).convert("RGBA"))
        arr = encode_segmap(arr, domain)
        arr = torch.from_numpy(arr).float()
        arr = arr.unsqueeze(0)
        return arr
    elif Path(path).suffix == ".npy":
        arr = np.load(path).astype(np.float32)  # .astype(np.uint8)
    elif is_image_file(path):
        arr = imread(path).astype(np.float32)  # .astype(np.uint8)
    else:
        raise ValueError("Unknown data type {}".format(path))

    # Convert from RGBA to RGB for images
    if len(arr.shape) == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, 0:3]
    if task == "x":
        arr = np.moveaxis(arr, 2, 0)

    if task == "m":
        arr[arr != 0] = 1
        # Make sure mask is single-channel
        if len(arr.shape) >= 3:
            arr = arr[:, :, 0]
        arr = np.expand_dims(arr, 0)

    return torch.from_numpy(arr).unsqueeze(0)


class OmniListDataset(Dataset):
    def __init__(self, mode, domain, opts, transform=None):

        self.domain = domain
        self.mode = mode
        self.tasks = set(opts.tasks)
        self.tasks.add("x")
        if "p" in self.tasks:
            self.tasks.add("m")

        file_list_path = Path(opts.data.files[mode][domain])
        if "/" not in str(file_list_path):
            file_list_path = Path(opts.data.files.base) / Path(
                opts.data.files[mode][domain]
            )

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        self.filter_samples()
        self.check_samples()
        self.file_list_path = str(file_list_path)
        self.transform = transform

    def filter_samples(self):
        """
        Filter out data which is not required for the model's tasks
        as defined in opts.tasks
        """
        self.samples_paths = [
            {k: v for k, v in s.items() if k in self.tasks} for s in self.samples_paths
        ]

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({
                domains: values
            }),
            paths: [{task: path}],
            domain: [domain],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """
        paths = self.samples_paths[i]

        # always apply transforms,
        # if no transform is specified, ToTensor and Normalize will be applied

        item = {
            "data": self.transform(
                {
                    task: tensor_loader(path, task, self.domain)
                    for task, path in paths.items()
                }
            ),
            "paths": paths,
            "domain": self.domain,
            "mode": self.mode,
        }
        if "d" in item["data"]:
            item["data"]["d"] = get_normalized_depth_t(item["data"]["d"], self.domain)

        return item

    def __len__(self):
        return len(self.samples_paths)

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"


def get_loader(mode, domain, opts):
    return DataLoader(
        OmniListDataset(
            mode, domain, opts, transform=transforms.Compose(get_transforms(opts))
        ),
        batch_size=opts.data.loaders.get("batch_size", 4),
        # shuffle=opts.data.loaders.get("shuffle", True),
        shuffle=True,
        num_workers=opts.data.loaders.get("num_workers", 8),
    )


def get_all_loaders(opts):
    loaders = {}
    for mode in ["train", "val"]:
        loaders[mode] = {}
        for domain in opts.domains:
            if mode in opts.data.files:
                if domain in opts.data.files[mode]:
                    loaders[mode][domain] = get_loader(mode, domain, opts)
    return loaders
