import torch.nn as nn
import torch.nn.functional as F


class NetSimCLR(nn.Module):
    def __init__(self, encoder, latent_shape, out_dim):
        super(NetSimCLR, self).__init__()
        self.base_encoder = encoder
        # projection MLP
        num_ftrs = latent_shape.numel()
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):  # h = representation, z = projection
        h = self.base_encoder(x)
        h = h.view(h.size(0), -1)

        z = self.l1(h)
        z = F.relu(z)
        z = self.l2(z)
        return h, z
