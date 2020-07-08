import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.discriminator import OmniDiscriminator, get_dis
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
    D: OmniDiscriminator = get_dis(opts, 0)
    loss = GANLoss().to(device)
    image = torch.rand(5, 3, 128, 128).to(torch.float32).to(device)
    image_adv = torch.rand(5, 2, 128, 128).to(torch.float32).to(device)
    # --------------------------------
    # -----  Test number params  -----
    # --------------------------------

    print(
        "Parameters in Discriminator: ", sum(p.numel() for p in D.parameters()),
    )

    # --------------------------
    # -----  Test Forward  -----
    # --------------------------
    for task, disc in D.items():
        for domain in disc.keys():
            disc.to(device)
            if domain == "Advent":
                d = disc[domain](image_adv)
            else:
                d = disc[domain](image)
            print(task, domain, d.shape, loss(d, True), loss(d, False))
