import sys
import torch

sys.path.append("..")


from run import tprint

from omnigan.utils import load_opts, domains_to_class_tensor
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy, l1_loss

if __name__ == "__main__":
    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")

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
