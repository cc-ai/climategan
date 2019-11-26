from pathlib import Path
from addict import Dict
import sys

sys.path.append("..")
from omnigan.discriminator import OmniDiscriminator
from omnigan.utils import load_conf

if __name__ == "__main__":
    conf = load_conf("../shared/defaults.yml")
    D = OmniDiscriminator(conf)

    print(D)

    print(
        "Parameters in each domain Discriminator: ",
        sum(p.numel() for p in D.T["n"].parameters()),
    )
