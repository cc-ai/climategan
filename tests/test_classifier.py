import sys
import torch
import numpy as np

sys.path.append("..")

from omnigan.utils import load_opts
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy

if __name__ == "__main__":
    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")

    rf_target = np.array([1, 0, 0, 0])
    rn_target = np.array([0, 1, 0, 0])
    sf_target = np.array([0, 0, 1, 0])
    sn_target = np.array([0, 0, 0, 1])

    labels = torch.from_numpy(np.array([rf_target, rn_target, sf_target, sn_target]))
    labels = torch.from_numpy(np.array([0, 1, 2, 3]))

    loss = cross_entropy()

    z = torch.ones(4, 128, 32, 32)
    latent_space = (128, 32, 32)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)

    print(y.shape)
    print(loss(y, labels))

    z = torch.ones(4, 256, 64, 64)
    latent_space = (256, 64, 64)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    print(y.shape)
    print(loss(y, labels))

    z = torch.ones(4, 64, 16, 16)
    latent_space = (64, 16, 16)
    C = get_classifier(opts, latent_space, 0)
    y = C(z)
    print(y.shape)
    print(loss(y, labels))
