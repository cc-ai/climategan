"""Define all losses. When possible, as inheriting from nn.Module
To send predictions to target.device
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from random import random as rand


class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        soft_shift=0.0,
        flip_prob=0.0,
        verbose=0,
    ):
        """Defines the GAN loss which uses either LSGAN or the regular GAN.
        When LSGAN is used, it is basically same as MSELoss,
        but it abstracts away the need to create the target label tensor
        that has the same size as the input +

        * label smoothing: target_real_label=0.75
        * label flipping: flip_prob > 0.

        source: https://github.com/sangwoomo/instagan/blob
        /b67e9008fcdd6c41652f8805f0b36bcaa8b632d6/models/networks.py

        Args:
            use_lsgan (bool, optional): Use MSE or BCE. Defaults to True.
            target_real_label (float, optional): Value for the real target.
                Defaults to 1.0.
            target_fake_label (float, optional): Value for the fake target.
                Defaults to 0.0.
            flip_prob (float, optional): Probability of flipping the label
                (use for real target in Discriminator only). Defaults to 0.0.
        """
        super().__init__()

        self.soft_shift = soft_shift
        self.verbose = verbose

        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.flip_prob = flip_prob

    def get_target_tensor(self, input, target_is_real):
        soft_change = torch.FloatTensor(1).uniform_(0, self.soft_shift)
        if self.verbose > 0:
            print("GANLoss sampled soft_change:", soft_change.item())
        if target_is_real:
            target_tensor = self.real_label - soft_change
        else:
            target_tensor = self.fake_label + soft_change
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if rand() < self.flip_prob:
            target_is_real = not target_is_real
            if self.verbose > 0:
                print("GANLoss: flipping label")
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.to(input.device))


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.to(logits.device))


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCELoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.to(logits.device))


class PixelCrossEntropy(CrossEntropy):
    """
    Computes the cross entropy per pixel
        in  > pred: b x c x h x w | label: b x h x w (int)
        out > b x h x w
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")


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


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target)
    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


class MSELoss(nn.Module):
    """
    Creates a criterion that measures the mean squared error
    (squared L2 norm) between each element in the input x and target y .
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, prediction, target):
        return self.loss(prediction, target.to(prediction.device))


class L1Loss(MSELoss):
    """
    Creates a criterion that measures the mean absolute error
    (MAE) between each element in the input x and target y
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()
