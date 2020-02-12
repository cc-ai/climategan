import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.discriminator import OmniDiscriminator
from omnigan.losses import GANLoss
from omnigan.utils import load_opts

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_opts(root / args.config, default=root / "shared/defaults.yaml")


if __name__ == "__main__":

    opts = opts.copy()

    D = OmniDiscriminator(opts)
    loss = GANLoss()

    print(D)

    print(
        "Parameters in each domain Discriminator: ",
        sum(p.numel() for p in D["t"]["n"].parameters()),
    )

    image = torch.from_numpy(np.random.rand(5, 3, 128, 128)).to(torch.float32)

    for task, disc in D.items():
        for domain in disc.keys():
            d = disc[domain](image)
            print(task, domain, d.shape, loss(d, True), loss(d, False))
