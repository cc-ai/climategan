from pathlib import Path
from addict import Dict
import sys

sys.path.append("..")
from omnigan.data import OmniListDataset, get_all_loaders
from omnigan.utils import load_opts, transforms_string

if __name__ == "__main__":

    opts = load_opts("../shared/defaults.yml")

    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True

    loaders = get_all_loaders(opts)

    ds = OmniListDataset(opts.data.files.train.rn)

    print(transforms_string(loaders.train.rn.dataset.transform))

    sample = ds[0]
    batch = Dict(next(iter(loaders.train.rn)))

    print("Batch: ", "items, ", " ".join(batch.keys()), "keys")

    for k in batch:
        print(
            k,
            batch[k].shape,
            batch[k].dtype,
            batch[k].min().item(),
            batch[k].max().item(),
        )
