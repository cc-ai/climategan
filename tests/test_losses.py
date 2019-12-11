import sys
import torch

sys.path.append("..")

from omnigan.data import get_all_loaders
from omnigan.utils import load_opts
from omnigan.generator import get_gen
from omnigan.losses import cross_entropy_2d
from run import print_header

if __name__ == "__main__":
    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    loaders = get_all_loaders(opts)
    batch = next(iter(loaders["train"]["rn"]))
    image = torch.randn(opts.data.loaders.batch_size, 3, 256, 256)
    G = get_gen(opts)
    z = G.encoder(image)

    print_header("test_crossentroy_2d")
    prediction = G.decoders["s"](z)
    print(cross_entropy_2d(prediction, batch["data"]["s"]))
