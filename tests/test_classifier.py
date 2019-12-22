import sys
import torch

sys.path.append("..")

from omnigan.utils import load_opts
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy, l1_loss

if __name__ == "__main__":
    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")

    labels = torch.LongTensor([0, 1, 2, 3])
    labels_float = torch.FloatTensor([0, 1, 2, 3])

    loss = cross_entropy()
    loss_l1 = l1_loss()

    z = torch.ones(4, 128, 32, 32)
    latent_space = (128, 32, 32)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)


    print(y.shape)
    print(loss(y, labels))
    print(loss_l1(y, labels_float))

    z = torch.ones(4, 256, 64, 64)
    latent_space = (256, 64, 64)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    print(y.shape)
    print(loss(y, labels))
    print(loss_l1(y, labels_float))

    z = torch.ones(4, 64, 16, 16)
    latent_space = (64, 16, 16)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    print(y.shape)
    print(loss(y, labels))
    print(loss_l1(y, labels_float))
