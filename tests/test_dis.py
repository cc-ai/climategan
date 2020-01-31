import sys
import torch
import numpy as np

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from omnigan.discriminator import OmniDiscriminator
from omnigan.losses import GANLoss
from run import opts

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
