import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.classifier import get_classifier
from omnigan.losses import CrossEntropy, L1Loss
from omnigan.tutils import domains_to_class_tensor
from omnigan.utils import load_test_opts
from run import tprint


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
    target_domains = ["rf", "rn", "sf", "sn", "rf"]
    labels = domains_to_class_tensor(target_domains, one_hot=False).to(device)
    one_hot_labels = domains_to_class_tensor(target_domains, one_hot=True).to(device)

    cross_entropy = CrossEntropy()
    loss_l1 = L1Loss()

    # ------------------------------
    # -----  Test C.forward()  -----
    # ------------------------------
    z = torch.ones(5, 128, 32, 32).to(device)
    latent_space = (128, 32, 32)
    C = get_classifier(opts, latent_space, 0).to(device)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    # --------------------------------
    # -----  Test cross_entropy  -----
    # --------------------------------
    tprint("CE loss:", cross_entropy(y, labels))
    # --------------------------
    # -----  Test l1_loss  -----
    # --------------------------
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()

    z = torch.ones(5, 256, 64, 64).to(device)
    # ------------------------------------------
    # -----  Test different latent shapes  -----
    # ------------------------------------------
    latent_space = (256, 64, 64)
    C = get_classifier(opts, latent_space, 0).to(device)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    tprint("CE loss:", cross_entropy(y, labels))
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()

    z = torch.ones(5, 64, 16, 16).to(device)
    latent_space = (64, 16, 16)
    C = get_classifier(opts, latent_space, 0).to(device)
    y = C(z)
    tprint(
        "output of classifier's shape for latent space {} :".format(list(z.shape[1:])),
        y.shape,
    )
    tprint("CE loss:", cross_entropy(y, labels))
    tprint("l1 loss:", loss_l1(y, one_hot_labels))
    print()
