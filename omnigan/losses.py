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


class TVLoss(nn.Module):
    """Total Variational Regularization: Penalizes differences in 
        neighboring pixel values

        source: https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
    """

    def __init__(self, tvloss_weight=1):
        """
        Args:
            TVLoss_weight (int, optional): [lambda i.e. weight for loss]. Defaults to 1.
        """
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tvloss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


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


def get_losses(opts, verbose):
    """Sets the loss functions to be used by G, D and C, as specified
    in the opts and returns a dictionnary of losses:

    losses = {
        "G": {
            "gan": {"a": ..., "t": ...},
            "cycle": {"a": ..., "t": ...}
            "auto": {"a": ..., "t": ...}
            "tasks": {"h": ..., "d": ..., "s": ..., etc.}
        },
        "D": GANLoss,
        "C": ...
    }
    """

    losses = {"G": {"a": {}, "t": {}, "tasks": {}}, "D": {}, "C": {}}

    # ------------------------------
    # -----  Generator Losses  -----
    # ------------------------------
    # translation losses
    if "a" in opts.tasks:
        losses["G"]["a"]["gan"] = GANLoss()
        losses["G"]["a"]["cycle"] = MSELoss()
        losses["G"]["a"]["auto"] = MSELoss()
        # ? add sm and dm losses too as in "t"
    if "t" in opts.tasks:
        losses["G"]["t"]["gan"] = GANLoss()
        losses["G"]["t"]["cycle"] = MSELoss()
        losses["G"]["t"]["auto"] = MSELoss()
        losses["G"]["t"]["sm"] = PixelCrossEntropy()
        losses["G"]["t"]["dm"] = MSELoss()
    # task losses
    # ? * add discriminator and gan loss to these task when no ground truth
    # ?   instead of noisy label
    if "d" in opts.tasks:
        losses["G"]["tasks"]["d"] = MSELoss()
    if "h" in opts.tasks:
        losses["G"]["tasks"]["h"] = MSELoss()
    if "s" in opts.tasks:
        losses["G"]["tasks"]["s"] = CrossEntropy()
    if "w" in opts.tasks:
        losses["G"]["tasks"]["w"] = lambda x, y: (x + y).mean()
    if "m" in opts.tasks:
        losses["G"]["tasks"]["m"] = {}
        losses["G"]["tasks"]["m"]["main"] = nn.BCELoss()
        losses["G"]["tasks"]["m"]["tv"] = TVLoss(opts.train.lambdas.G.m.tv)

    # undistinguishable features loss
    # TODO setup a get_losses func to assign the right loss according to the yaml
    if opts.classifier.loss == "l1":
        loss_classifier = L1Loss()
    elif opts.classifier.loss == "l2":
        loss_classifier = MSELoss()
    else:
        loss_classifier = CrossEntropy()
    losses["G"]["classifier"] = loss_classifier
    # -------------------------------
    # -----  Classifier Losses  -----
    # -------------------------------
    losses["C"] = loss_classifier
    # ----------------------------------
    # -----  Discriminator Losses  -----
    # ----------------------------------
    losses["D"] = GANLoss(
        soft_shift=opts.dis.soft_shift, flip_prob=opts.dis.flip_prob, verbose=verbose,
    )
    return losses


def prob_2_entropy(prob):
    """ 
    convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

# class CrossEntropyLoss(nn.Module):
#     """
#     Calculate the cross entropy loss by function cross_entropy_2d
#     """
#     # Almost the same as cross_entropy_2d, I just integrate .long().to(prediction.device) on the target
#     def __init__(self):
#         super().__init__()
#         self.loss = cross_entropy_2d

#     def __call__(self, prediction, target):
#         return self.loss(prediction, target.long().to(prediction.device))


class BCELoss(nn.Module):
    # Almost the same as BCEWithLogitsLoss
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target):
        return self.loss(prediction, torch.FloatTensor(prediction.size()).
                        fill_(target).to(prediction.get_device()))


class ADVENTSegLoss(nn.Module):
    def __init__(
        self,
        opts,
    ):
        super().__init__()
        self.opts = opts
        self.loss = cross_entropy_2d

    def __call__(self, prediction1, prediction2, target):
        if self.opts.dis.ADVENT.multi_level == True:
            loss_seg_src_aux = self.loss(prediction1, target.long().to(prediction1.device))
        else:
            loss_seg_src_aux = 0
        loss_seg_src_main = self.loss(prediction2, target.long().to(prediction2.device))
        
        loss = (self.opts.train.lambdas.advent.seg_main * loss_seg_src_main
                + self.opts.train.lambdas.advent.seg_aux * loss_seg_src_aux)
        
        return loss


class ADVENTAdversarialLoss(nn.Module):
    def __init__(
        self,
        opts
    ):
        super().__init__()
        self.opts = opts
        self.loss = BCELoss()
    
    def __call__(self, prediction1, prediction2, target, discriminator_main, discriminator_aux=None):
        
        if self.opts.dis.ADVENT.multi_level == True:
            d_out_aux = discriminator_aux(prob_2_entropy(F.softmax(prediction1, dim = 1)))
            loss_adv_trg_aux = self.loss(d_out_aux, target)
        else:
            loss_adv_trg_aux = 0
        d_out_main = discriminator_main(prob_2_entropy(F.softmax(prediction2, dim = 1)))
        loss_adv_trg_main = self.loss(d_out_main, target)
        
        loss = (self.opts.train.lambdas.advent.adv_main * loss_adv_trg_main
                + self.opts.train.lambdas.advent.adv_aux * loss_adv_trg_aux)
        
        return loss
