import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import get_all_loaders
from omnigan.generator import get_gen
from omnigan.losses import cross_entropy_2d
from omnigan.utils import load_opts
from run import print_header


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_opts(root / args.config, default=root / "shared/defaults.yaml")


if __name__ == "__main__":
    opts = opts.copy()
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    loaders = get_all_loaders(opts)
    batch = next(iter(loaders["train"]["rn"]))
    image = torch.randn(opts.data.loaders.batch_size, 3, 256, 256)
    G = get_gen(opts)
    z = G.encode(image)

    print_header("test_crossentroy_2d")
    prediction = G.decoders["s"](z)
    print(cross_entropy_2d(prediction, batch["data"]["s"]))
    # ! error how to infer from cropped data: input: 224 output: 256??
