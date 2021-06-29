"""Data-loading functions in order to create a Dataset and DataLoaders.
Transforms for loaders are in transforms.py
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from imageio import imread
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from climategan.transforms import get_transforms
from climategan.tutils import get_normalized_depth_t
from climategan.utils import env_to_path, is_image_file

classes_dict = {
    "s": {  # unity
        0: [0, 0, 255, 255],  # Water
        1: [55, 55, 55, 255],  # Ground
        2: [0, 255, 255, 255],  # Building
        3: [255, 212, 0, 255],  # Traffic items
        4: [0, 255, 0, 255],  # Vegetation
        5: [255, 97, 0, 255],  # Terrain
        6: [255, 0, 0, 255],  # Car
        7: [60, 180, 60, 255],  # Trees
        8: [255, 0, 255, 255],  # Person
        9: [0, 0, 0, 255],  # Sky
        10: [255, 255, 255, 255],  # Default
    },
    "r": {  # deeplab v2
        0: [0, 0, 255, 255],  # Water
        1: [55, 55, 55, 255],  # Ground
        2: [0, 255, 255, 255],  # Building
        3: [255, 212, 0, 255],  # Traffic items
        4: [0, 255, 0, 255],  # Vegetation
        5: [255, 97, 0, 255],  # Terrain
        6: [255, 0, 0, 255],  # Car
        7: [60, 180, 60, 255],  # Trees
        8: [220, 20, 60, 255],  # Person
        9: [8, 19, 49, 255],  # Sky
        10: [0, 80, 100, 255],  # Default
    },
    "kitti": {
        0: [210, 0, 200],  # Terrain
        1: [90, 200, 255],  # Sky
        2: [0, 199, 0],  # Tree
        3: [90, 240, 0],  # Vegetation
        4: [140, 140, 140],  # Building
        5: [100, 60, 100],  # Road
        6: [250, 100, 255],  # GuardRail
        7: [255, 255, 0],  # TrafficSign
        8: [200, 200, 0],  # TrafficLight
        9: [255, 130, 0],  # Pole
        10: [80, 80, 80],  # Misc
        11: [160, 60, 60],  # Truck
        12: [255, 127, 80],  # Car
        13: [0, 139, 139],  # Van
        14: [0, 0, 0],  # Undefined
    },
    "flood": {
        0: [255, 0, 0],  # Cannot flood
        1: [0, 0, 255],  # Must flood
        2: [0, 0, 0],  # May flood
    },
}

kitti_mapping = {
    0: 5,  # Terrain -> Terrain
    1: 9,  # Sky -> Sky
    2: 7,  # Tree -> Trees
    3: 4,  # Vegetation ->  Vegetation
    4: 2,  # Building -> Building
    5: 1,  # Road -> Ground
    6: 3,  # GuardRail -> Traffic items
    7: 3,  # TrafficSign ->  Traffic items
    8: 3,  # TrafficLight -> Traffic items
    9: 3,  # Pole -> Traffic items
    10: 10,  # Misc -> default
    11: 6,  # Truck -> Car
    12: 6,  # Car -> Car
    13: 6,  # Van -> Car
    14: 10,  # Undefined -> Default
}


def encode_exact_segmap(seg, classes_dict, default_value=14):
    """
    When the mapping (rgb -> label) is known to be exact (no approximative rgb values)
    maps rgb image to segmap labels

    Args:
        seg (np.ndarray): H x W x 3 RGB image
        classes_dict (dict): Mapping {class: rgb value}
        default_value (int, optional): Value for unknown label. Defaults to 14.

    Returns:
        np.ndarray: Segmap as labels, not RGB
    """
    out = np.ones((seg.shape[0], seg.shape[1])) * default_value
    for cindex, cvalue in classes_dict.items():
        out[np.where((seg == cvalue).all(-1))] = cindex
    return out


def merge_labels(labels, mapping, default_value=14):
    """
    Maps labels from a source domain to labels of a target domain,
    typically kitti -> climategan

    Args:
        labels (np.ndarray): input segmap labels
        mapping (dict): source_label -> target_label
        default_value (int, optional): Unknown label. Defaults to 14.

    Returns:
        np.ndarray: Adapted labels
    """
    out = np.ones_like(labels) * default_value
    for source, target in mapping.items():
        out[labels == source] = target
    return out


def process_kitti_seg(path, kitti_classes, merge_map, default=14):
    """
    Processes a path to produce a 1 x 1 x H x W torch segmap

    %timeit process_kitti_seg(path, classes_dict, mapping, default=14)
    326 ms ± 118 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    Args:
        path (str | pathlib.Path): Segmap RBG path
        kitti_classes (dict): Kitti map label -> rgb
        merge_map (dict): map kitti_label -> climategan_label
        default (int, optional): Unknown kitti label. Defaults to 14.

    Returns:
        torch.Tensor: 1 x 1 x H x W torch segmap
    """
    seg = imread(path)
    labels = encode_exact_segmap(seg, kitti_classes, default_value=default)
    merged = merge_labels(labels, merge_map, default_value=default)
    return torch.tensor(merged).unsqueeze(0).unsqueeze(0)


def decode_segmap_merged_labels(tensor, domain, is_target, nc=11):
    """Creates a label colormap for classes used in Unity segmentation benchmark.
    Arguments:
        tensor -- segmented image of size (1) x (nc) x (H) x (W)
            if prediction, or size (1) x (1) x (H) x (W) if target
    Returns:
        RGB tensor of size (1) x (3) x (H) x (W)
    #"""

    if is_target:  # Target is size 1 x 1 x H x W
        idx = tensor.squeeze(0).squeeze(0)
    else:  # Prediction is size 1 x nc x H x W
        idx = torch.argmax(tensor.squeeze(0), dim=0)

    indexer = torch.tensor(list(classes_dict[domain].values()))[:, :3]
    return indexer[idx.long()].permute(2, 0, 1).to(torch.float32).unsqueeze(0)


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

    for col in range(nc):
        idx = image == col
        r[idx] = colormap[col, 0]
        g[idx] = colormap[col, 1]
        b[idx] = colormap[col, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def find_closest_class(pixel, dict_classes):
    """Takes a pixel as input and finds the closest known pixel value corresponding
    to a class in dict_classes

    Arguments:
        pixel -- tuple pixel (R,G,B,A)
    Returns:
        tuple pixel (R,G,B,A) corresponding to a key (a class) in dict_classes
    """
    min_dist = float("inf")
    closest_pixel = None
    for pixel_value in dict_classes.keys():
        dist = np.sqrt(np.sum(np.square(np.subtract(pixel, pixel_value))))
        if dist < min_dist:
            min_dist = dist
            closest_pixel = pixel_value
    return closest_pixel


def encode_segmap(arr, domain):
    """Change a segmentation RGBA array to a segmentation array
                            with each pixel being the index of the class
    Arguments:
        numpy array -- segmented image of size (H) x (W) x (4 RGBA values)
    Returns:
        numpy array of size (1) x (H) x (W) with each pixel being the index of the class
    """
    new_arr = np.zeros((1, arr.shape[0], arr.shape[1]))
    dict_classes = {
        tuple(rgba_value): class_id
        for (class_id, rgba_value) in classes_dict[domain].items()
    }
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            pixel_rgba = tuple(arr[i, j, :])
            if pixel_rgba in dict_classes.keys():
                new_arr[0, i, j] = dict_classes[pixel_rgba]
            else:
                pixel_rgba_closest = find_closest_class(pixel_rgba, dict_classes)
                new_arr[0, i, j] = dict_classes[pixel_rgba_closest]
    return new_arr


def encode_mask_label(arr, domain):
    """Change a segmentation RGBA array to a segmentation array
                            with each pixel being the index of the class
    Arguments:
        numpy array -- segmented image of size (H) x (W) x (3 RGB values)
    Returns:
        numpy array of size (1) x (H) x (W) with each pixel being the index of the class
    """
    diff = np.zeros((len(classes_dict[domain].keys()), arr.shape[0], arr.shape[1]))
    for cindex, cvalue in classes_dict[domain].items():
        diff[cindex, :, :] = np.sqrt(
            np.sum(
                np.square(arr - np.tile(cvalue, (arr.shape[0], arr.shape[1], 1))),
                axis=2,
            )
        )
    return np.expand_dims(np.argmin(diff, axis=0), axis=0)


def transform_segmap_image_to_tensor(path, domain):
    """
    Transforms a segmentation image to a tensor of size (1) x (1) x (H) x (W)
    with each pixel being the index of the class
    """
    arr = np.array(Image.open(path).convert("RGBA"))
    arr = encode_segmap(arr, domain)
    arr = torch.from_numpy(arr).float()
    arr = arr.unsqueeze(0)
    return arr


def save_segmap_tensors(path_to_json, path_to_dir, domain):
    """
    Loads the segmentation images mentionned in a json file, transforms them to
    tensors and save the tensors in the wanted directory

    Args:
        path_to_json: complete path to the json file where to find the original data
        path_to_dir: path to the directory where to save the tensors as tensor_name.pt
        domain: domain of the images ("r" or "s")

    e.g:
        save_tensors(
            "/network/tmp1/ccai/data/climategan/seg/train_s.json",
            "/network/tmp1/ccai/data/munit_dataset/simdata/Unity11K_res640/Seg_tensors/",
            "s",
        )
    """
    ims_list = None
    if path_to_json:
        path_to_json = Path(path_to_json).resolve()
        with open(path_to_json, "r") as f:
            ims_list = yaml.safe_load(f)

    assert ims_list is not None

    for im_dict in ims_list:
        for task_name, path in im_dict.items():
            if task_name == "s":
                file_name = os.path.splitext(path)[0]  # remove extension
                file_name = file_name.rsplit("/", 1)[-1]  # keep only the file_name
                tensor = transform_segmap_image_to_tensor(path, domain)
                torch.save(tensor, path_to_dir + file_name + ".pt")


def pil_image_loader(path, task):
    if Path(path).suffix == ".npy":
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
        arr[arr != 0] = 1
        # Make sure mask is single-channel
        if len(arr.shape) >= 3:
            arr = arr[:, :, 0]

    # assert len(arr.shape) == 3, (path, task, arr.shape)

    return Image.fromarray(arr)


def tensor_loader(path, task, domain, opts):
    """load data as tensors
    Args:
        path (str): path to data
        task (str)
        domain (str)
    Returns:
        [Tensor]: 1 x C x H x W
    """
    if task == "s":
        if domain == "kitti":
            return process_kitti_seg(
                path, classes_dict["kitti"], kitti_mapping, default=14
            )
        return torch.load(path)
    elif task == "d":
        if Path(path).suffix == ".npy":
            arr = np.load(path)
        else:
            arr = imread(path)  # .astype(np.uint8) /!\ kitti is np.uint16
        tensor = torch.from_numpy(arr.astype(np.float32))
        tensor = get_normalized_depth_t(
            tensor,
            domain,
            normalize="d" in opts.train.pseudo.tasks,
            log=opts.gen.d.classify.enable,
        )
        tensor = tensor.unsqueeze(0)
        return tensor

    elif Path(path).suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif is_image_file(path):
        arr = imread(path).astype(np.float32)
    else:
        raise ValueError("Unknown data type {}".format(path))

    # Convert from RGBA to RGB for images
    if len(arr.shape) == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, 0:3]

    if task == "x":
        arr -= arr.min()
        arr /= arr.max()
        arr = np.moveaxis(arr, 2, 0)
    elif task == "s":
        arr = np.moveaxis(arr, 2, 0)
    elif task == "m":
        if arr.max() > 127:
            arr = (arr > 127).astype(arr.dtype)
        # Make sure mask is single-channel
        if len(arr.shape) >= 3:
            arr = arr[:, :, 0]
        arr = np.expand_dims(arr, 0)

    return torch.from_numpy(arr).unsqueeze(0)


class OmniListDataset(Dataset):
    def __init__(self, mode, domain, opts, transform=None):

        self.opts = opts
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

        if opts.data.max_samples and opts.data.max_samples != -1:
            assert isinstance(opts.data.max_samples, int)
            self.samples_paths = self.samples_paths[: opts.data.max_samples]

        self.filter_samples()
        if opts.data.check_samples:
            print(f"Checking samples ({mode}, {domain})")
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
                    task: tensor_loader(
                        env_to_path(path),
                        task,
                        self.domain,
                        self.opts,
                    )
                    for task, path in paths.items()
                }
            ),
            "paths": paths,
            "domain": self.domain if self.domain != "kitti" else "s",
            "mode": self.mode,
        }

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
    if (
        domain != "kitti"
        or not opts.train.kitti.pretrain
        or not opts.train.kitti.batch_size
    ):
        batch_size = opts.data.loaders.get("batch_size", 4)
    else:
        batch_size = opts.train.kitti.get("batch_size", 4)

    return DataLoader(
        OmniListDataset(
            mode,
            domain,
            opts,
            transform=transforms.Compose(get_transforms(opts, mode, domain)),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=opts.data.loaders.get("num_workers", 8),
        pin_memory=True,  # faster transfer to gpu
        drop_last=True,  # avoids batchnorm pbs if last batch has size 1
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
