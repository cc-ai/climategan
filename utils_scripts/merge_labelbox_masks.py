from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from shutil import copyfile

if __name__ == "__main__":
    # output of download_labelbox.py
    base_dir = Path("/Users/victor/Downloads/labelbox_test_flood-v2")
    labeled_dir = base_dir / "__labeled"
    assert base_dir.exists()
    labeled_dir.mkdir(exist_ok=True)

    sub_dirs = [
        d
        for d in base_dir.expanduser().resolve().iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__labeled"
    ]

    for k, sd in enumerate(sub_dirs):
        print(k + 1, "/", len(sub_dirs), sd.name)

        # must-flood binary mask
        must = np.stack([imread(i)[:, :, :3] for i in sd.glob("*must*.png")]).sum(0) > 0
        # cannot-flood binary mask
        cannot = (
            np.stack([imread(i)[:, :, :3] for i in sd.glob("*cannot*.png")]).sum(0) > 0
        )
        # must is red
        must = (must * [0, 0, 255]).astype(np.uint8)
        # connot is blue
        cannot = (cannot * [255, 0, 0]).astype(np.uint8)
        # merged labels
        label = must + cannot
        # check no overlap
        assert sorted(np.unique(label)) == [0, 255]
        # create filename
        stem = "_".join(list(sd.glob("*must*.png"))[0].stem.split("_")[:-2])
        # save label
        imsave(sd / f"{stem}_labeled.png", label)
        copyfile(sd / f"{stem}_labeled.png", labeled_dir / f"{stem}_labeled.png")
