import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import get_all_loaders
from omnigan.generator import get_gen
from omnigan.losses import PixelCrossEntropy
from omnigan.utils import load_test_opts
from run import print_header


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_all_loaders(opts)
    batch = next(iter(loaders["train"]["r"]))
    image = torch.randn(opts.data.loaders.batch_size, 3, 32, 32).to(device)
    G = get_gen(opts).to(device)
    z = G.encode(image)

    # -----------------------------------
    # -----  Test cross_entropy_2d  -----
    # -----------------------------------
    print_header("test_crossentroy_2d")
    prediction = G.decoders["s"](z)
    pce = PixelCrossEntropy()
    print(pce(prediction.squeeze(), batch["data"]["s"].long().squeeze().to(device)))
    # ! error how to infer from cropped data: input: 224 output: 256??

    # TODO more test for the losses
