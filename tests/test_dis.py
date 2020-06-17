import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.discriminator import OmniDiscriminator
from omnigan.losses import GANLoss
from omnigan.utils import load_test_opts

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = OmniDiscriminator(opts).to(device)
    loss = GANLoss().to(device)
    image = torch.rand(5, 3, 128, 128).to(torch.float32).to(device)

    # --------------------------
    # -----  Test Forward  -----
    # --------------------------
    for task, disc in D.items():
        if not isinstance(disc, nn.ModuleDict):
            disc = {"": disc}
        for domain in disc.keys():
            print(
                f"Parameters in discriminator {task} {domain}: ",
                sum(p.numel() for p in disc[domain].parameters()),
            )
            d = disc[domain](image)
            print(task, domain, d.shape, loss(d, True), loss(d, False))
