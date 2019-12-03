import sys
import torch
import numpy as np

sys.path.append("..")

from omnigan.discriminator import OmniDiscriminator
from omnigan.utils import load_opts
from omnigan.losses import GANLoss

if __name__ == "__main__":
    opts = load_opts("../shared/defaults.yml")
    D = OmniDiscriminator(opts)
    loss = GANLoss()

    print(D)

    print(
        "Parameters in each domain Discriminator: ",
        sum(p.numel() for p in D.models["t"]["n"].parameters()),
    )

    image = torch.from_numpy(np.random.rand(5, 3, 512, 512)).to(torch.float32)

    for task, disc in D.models.items():
        for domain in disc.keys():
            d = disc[domain](image)
            print(task, domain, d.shape, loss(d, True), loss(d, False))
