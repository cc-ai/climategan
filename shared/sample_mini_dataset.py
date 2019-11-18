from pathlib import Path
from numpy.random import permutation
from shutil import copyfile, rmtree

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

if __name__ == "__main__":

    train_size = 170
    val_size = 30

    root = Path("/network/tmp1/ccai/data/munit_dataset")
    train_flooded = root / "mini" / "train" / "flooded"
    train_non_flooded = root / "mini" / "train" / "non_flooded"
    val_flooded = root / "mini" / "val" / "flooded"
    val_non_flooded = root / "mini" / "val" / "non_flooded"

    flooded_source = root / "flooded" / "imgs_png"
    non_flooded_source = root / "non_flooded" / "streetview_mvp"

    assert "/mini/" in str(train_flooded)
    assert "/mini/" in str(train_non_flooded)
    assert "/mini/" in str(val_flooded)
    assert "/mini/" in str(val_non_flooded)

    rmtree(train_flooded, ignore_errors=True)
    rmtree(train_non_flooded, ignore_errors=True)
    rmtree(val_flooded, ignore_errors=True)
    rmtree(val_non_flooded, ignore_errors=True)

    train_flooded.mkdir(exist_ok=True, parents=True)
    train_non_flooded.mkdir(exist_ok=True, parents=True)
    val_non_flooded.mkdir(exist_ok=True, parents=True)
    val_flooded.mkdir(exist_ok=True, parents=True)

    flooded_ims_source = [
        im
        for im in flooded_source.iterdir()
        if im.is_file() and im.suffix in IMG_EXTENSIONS
    ]
    perm = permutation(len(flooded_ims_source))[: train_size + val_size]
    flooded_ims = [flooded_ims_source[i] for i in perm]

    non_flooded_ims_source = [
        im
        for im in non_flooded_source.iterdir()
        if im.is_file() and im.suffix in IMG_EXTENSIONS
    ]
    perm = permutation(len(non_flooded_ims_source))[: train_size + val_size]
    non_flooded_ims = [non_flooded_ims_source[i] for i in perm]

    print("Found {} images".format(train_size + val_size))

    for i, im in enumerate(flooded_ims):
        if i < train_size:
            copyfile(im, train_flooded / im.name)
        else:
            copyfile(im, val_flooded / im.name)

    print("Copied flooded images")

    for i, im in enumerate(non_flooded_ims):
        if i < train_size:
            copyfile(im, train_non_flooded / im.name)
        else:
            copyfile(im, val_non_flooded / im.name)

    print("Copied non_flooded images")
