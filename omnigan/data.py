from pathlib import Path
import yaml
import json
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from torchvision import transforms
import numpy as np
from .transforms import get_transforms
from PIL import Image

# ? paired dataset

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


def decode_segmap(image, nc=19):
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


def is_image_file(filename):
    """Check that a file's name points to a known image format
    """
    return Path(filename).suffix in IMG_EXTENSIONS


def pil_image_loader(path, task):
    if Path(path).suffix == ".npy":
        arr = np.load(path).astype(np.uint8)
    elif is_image_file(path):
        arr = imread(path).astype(np.uint8)
    else:
        raise ValueError("Unknown data type {}".format(path))

    if task == "d":
        arr = arr.astype(np.float32)
        arr[arr != 0] = 1 / arr[arr != 0]

    if task == "s":
        arr = decode_segmap(arr)

    # assert len(arr.shape) == 3, (path, task, arr.shape)

    # pdb.set_trace()

    return Image.fromarray(arr)


class OmniListDataset(Dataset):
    def __init__(self, mode, domain, opts, transform=None):

        self.domain = domain
        self.mode = mode

        file_list_path = Path(opts.data.files[mode][domain])

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        self.check_samples()
        self.file_list_path = str(file_list_path)
        self.transform = transform

    def __getitem__(self, i):
        if self.transform:
            return {
                "data": self.transform(
                    {
                        task: pil_image_loader(path, task)
                        for task, path in self.samples_paths[i].items()
                    }
                ),
                "paths": self.samples_paths[i],
                "domain": self.domain,
                "mode": self.mode,
            }

        return {
            "data": {
                task: pil_image_loader(path, task)
                for task, path in self.samples_paths[i].items()
            },
            "paths": self.samples_paths[i],
            "domain": self.domain,
            "mode": self.mode,
        }

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


def get_loader(domain, mode, opts):
    return DataLoader(
        OmniListDataset(
            domain, mode, opts, transform=transforms.Compose(get_transforms(opts)),
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
        for domain in ["rf", "rn", "sf", "sn"]:
            if mode in opts.data.files:
                if domain in opts.data.files[mode]:
                    loaders[mode][domain] = get_loader(mode, domain, opts)
    return loaders
