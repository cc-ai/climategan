from torch import nn
import numpy as np

from omnigan.utils import init_weights


def get_classifier(opts, latent_shape):
    latent_size = np.prod(latent_shape)
    C = OmniClassifier(opts, latent_size)
    init_weights(
        C, init_type=opts.classifier.init_type, init_gain=opts.classifier.init_gain
    )
    return C


class OmniClassifier(nn.Module):
    def __init__(self, opts, latent_size):
        super().__init__()
        self.opts = opts
        self.latent_size = latent_size
        prev = latent_size
        self.model = []
        for i, l in enumerate(opts.classifier.layers):
            layer = [nn.Linear(prev, l)]
            if i != len(opts.classifier.layers) - 1:
                layer += [
                    nn.LeakyReLU(),
                    nn.Dropout(opts.classifier.dropout),
                ]
            self.model += layer
            prev = l
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)
