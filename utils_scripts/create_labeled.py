from pathlib import Path
from skimage.io import imread, imsave
import numpy as np

if __name__ == "__main__":
    impath = Path("/Users/victor/Downloads/metrics-v2/imgs")
    labpath = Path("/Users/victor/Downloads/metrics-v2/labels")
    outpath = Path("/Users/victor/Downloads/metrics-v2/labeled")
    outpath.mkdir(exist_ok=True, parents=True)
    ims = sorted(
        [d for d in impath.iterdir() if d.is_file() and not d.name.startswith(".")],
        key=lambda x: x.stem,
    )
    labs = sorted(
        [d for d in labpath.iterdir() if d.is_file() and not d.name.startswith(".")],
        key=lambda x: x.stem.replace("_labeled", ""),
    )

    for k, (i, l) in enumerate(zip(ims, labs)):
        print(f"{k + 1} / {len(ims)}", end="\r", flush=True)
        assert i.stem == l.stem.replace("_labeled", "")
        im = imread(i)[:, :, :3]
        la = imread(l)
        ld = (0.7 * im + 0.3 * la).astype(np.uint8)
        imsave(outpath / i.name, ld)
