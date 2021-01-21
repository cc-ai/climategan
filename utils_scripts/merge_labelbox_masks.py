from pathlib import Path
import numpy as np
from skimage.io import imread, imsave

if __name__ == "__main__":
    base_dir = Path("/Users/victor/Downloads/labelbox_test_flood")
    assert base_dir.exists()

    sub_dirs = [
        d
        for d in base_dir.expanduser().resolve().iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for k, sd in enumerate(sub_dirs):
        print(k + 1, "/", len(sub_dirs), sd.name)
        must = np.stack([imread(i)[:, :, :3] for i in sd.glob("*must*.png")]).sum(0) > 0
        cannot = (
            np.stack([imread(i)[:, :, :3] for i in sd.glob("*cannot*.png")]).sum(0) > 0
        )
        stem = "_".join(list(sd.glob("*must*.png"))[0].stem.split("_")[:-2])
        must = (must * [0, 0, 255]).astype(np.uint8)
        cannot = (cannot * [255, 0, 0]).astype(np.uint8)
        label = must + cannot
        assert sorted(np.unique(label)) == [0, 255]
        imsave(sd / f"{stem}_labeled.png", label)
