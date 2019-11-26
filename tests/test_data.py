from pathlib import Path
from addict import Dict
import sys

sys.path.append("..")
import omnigan

if __name__ == "__main__":

    example_path = Path() / "../example_data"
    opts = Dict()
    opts.data.files.train.rn = str(example_path / "train_rn.json")
    opts.data.files.val.rn = str(example_path / "val_rn.json")
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    opts.data.transforms = [Dict()]
    opts.data.transforms[0].name = "hflip"
    opts.data.transforms[0].p = 0.5

    loaders = omnigan.data.get_all_loaders(opts)
    ds = omnigan.data.OmniListDataset(opts.data.files.train.rn)

    sample = next(iter(ds))
    batch = Dict(next(iter(loaders.train.rn)))
