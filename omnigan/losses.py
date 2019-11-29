import torch
import torch.nn as nn


# source: https://github.com/sangwoomo/instagan/blob/b67e9008fcdd6c41652f8805f0b36bcaa8b632d6/models/networks.py
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def cross_entropy():
    # loss(logits, target) with target as classes, NOT 1-h encoded
    return torch.nn.CrossEntropyLoss()


class TravelLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def cosine_loss(self, real, fake):
        norm_real = torch.norm(real, p=2, dim=1)[:, None]
        norm_fake = torch.norm(fake, p=2, dim=1)[:, None]
        mat_real = real / norm_real
        mat_fake = fake / norm_fake
        mat_real = torch.max(mat_real, self.eps * torch.ones_like(mat_real))
        mat_fake = torch.max(mat_fake, self.eps * torch.ones_like(mat_fake))
        # compute only the diagonal of the matrix multiplication
        return torch.einsum("ij, ji -> i", mat_fake, mat_real).sum()

    def __call__(self, S_real, S_fake):
        self.v_real = []
        self.v_fake = []
        for i in range(len(S_real)):
            for j in range(i):
                self.v_real.append((S_real[i] - S_real[j])[None, :])
                self.v_fake.append((S_fake[i] - S_fake[j])[None, :])
        self.v_real_t = torch.cat(self.v_real, dim=0)
        self.v_fake_t = torch.cat(self.v_fake, dim=0)
        return self.cosine_loss(self.v_real_t, self.v_fake_t)
