import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy, l1_loss
from omnigan.utils import domains_to_class_tensor, load_opts
from run import tprint


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/local_tests.yaml")
args = parser.parse_args()

root = Path(__file__).parent.parent
opts = load_opts(root / args.config, default=root / "shared/defaults.yaml")


if __name__ == "__main__":

    opts = opts.copy()

    target_domains = ["rf", "rn", "sf", "sn", "rf"]
    labels = domains_to_class_tensor(target_domains, one_hot=False)
    one_hot_labels = domains_to_class_tensor(target_domains, one_hot=True)

    cross_entropy = cross_entropy()
    loss_l1 = l1_loss()

    z = torch.ones(5, 128, 32, 32)
    latent_space = (128, 32, 32)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    tprint("CE loss:", cross_entropy(y, labels))
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()

    z = torch.ones(5, 256, 64, 64)
    latent_space = (256, 64, 64)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    tprint("CE loss:", cross_entropy(y, labels))
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()

    z = torch.ones(5, 64, 16, 16)
    latent_space = (64, 16, 16)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    tprint("CE loss:", cross_entropy(y, labels))
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()
