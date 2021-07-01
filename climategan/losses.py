"""Define all losses. When possible, as inheriting from nn.Module
To send predictions to target.device
"""
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
            self.loss = nn.BCEWithLogitsLoss()
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

    def __call__(self, input, target_is_real, *args, **kwargs):
        r = rand()
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                if r < self.flip_prob:
                    target_is_real = not target_is_real
                target_tensor = self.get_target_tensor(pred_i, target_is_real)
                loss_tensor = self.loss(pred_i, target_tensor.to(pred_i.device))
                loss += loss_tensor
            return loss / len(input)
        else:
            if r < self.flip_prob:
                target_is_real = not target_is_real
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor.to(input.device))


class FeatMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterionFeat = nn.L1Loss()

    def __call__(self, pred_real, pred_fake):
        # pred_{real, fake} are lists of features
        num_D = len(pred_fake)
        GAN_Feat_loss = 0.0
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach()
                )
                GAN_Feat_loss += unweighted_loss / num_D
        return GAN_Feat_loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.to(logits.device).long())


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

    source:
    https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
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


class MinentLoss(nn.Module):
    """
    Loss for the minimization of the entropy map
    Source for version 1: https://github.com/valeoai/ADVENT

    Version 2 adds the variance of the entropy map in the computation of the loss
    """

    def __init__(self, version=1, lambda_var=0.1):
        super().__init__()
        self.version = version
        self.lambda_var = lambda_var

    def __call__(self, pred):
        assert pred.dim() == 4
        n, c, h, w = pred.size()
        entropy_map = -torch.mul(pred, torch.log2(pred + 1e-30)) / np.log2(c)
        if self.version == 1:
            return torch.sum(entropy_map) / (n * h * w)
        else:
            entropy_map_demean = entropy_map - torch.sum(entropy_map) / (n * h * w)
            entropy_map_squ = torch.mul(entropy_map_demean, entropy_map_demean)
            return torch.sum(entropy_map + self.lambda_var * entropy_map_squ) / (
                n * h * w
            )


class MSELoss(nn.Module):
    """
    Creates a criterion that measures the mean squared error
    (squared L2 norm) between each element in the input x and target y .
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target):
        return self.loss(prediction, target.to(prediction.device))


class L1Loss(MSELoss):
    """
    Creates a criterion that measures the mean absolute error
    (MAE) between each element in the input x and target y
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()


class SIMSELoss(nn.Module):
    """Scale invariant MSE Loss"""

    def __init__(self):
        super(SIMSELoss, self).__init__()

    def __call__(self, prediction, target):
        d = prediction - target
        diff = torch.mean(d * d)
        relDiff = torch.mean(d) * torch.mean(d)
        return diff - relDiff


class SIGMLoss(nn.Module):
    """loss from MiDaS paper
    MiDaS did not specify how the gradients were computed but we use Sobel
    filters which approximate the derivative of an image.
    """

    def __init__(self, gmweight=0.5, scale=4, device="cuda"):
        super(SIGMLoss, self).__init__()
        self.gmweight = gmweight
        self.sobelx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(device)
        self.sobely = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(device)
        self.scale = scale

    def __call__(self, prediction, target):
        # get disparities
        # align both the prediction and the ground truth to have zero
        # translation and unit scale
        t_pred = torch.median(prediction)
        t_targ = torch.median(target)
        s_pred = torch.mean(torch.abs(prediction - t_pred))
        s_targ = torch.mean(torch.abs(target - t_targ))
        pred = (prediction - t_pred) / s_pred
        targ = (target - t_targ) / s_targ

        R = pred - targ

        # get gradient map with sobel filters
        batch_size = prediction.size()[0]
        num_pix = prediction.size()[-1] * prediction.size()[-2]
        sobelx = (self.sobelx).expand((batch_size, 1, -1, -1))
        sobely = (self.sobely).expand((batch_size, 1, -1, -1))
        gmLoss = 0  # gradient matching term
        for k in range(self.scale):
            R_ = F.interpolate(R, scale_factor=1 / 2 ** k)
            Rx = F.conv2d(R_, sobelx, stride=1)
            Ry = F.conv2d(R_, sobely, stride=1)
            gmLoss += torch.sum(torch.abs(Rx) + torch.abs(Ry))
        gmLoss = self.gmweight / num_pix * gmLoss
        # scale invariant MSE
        simseLoss = 0.5 / num_pix * torch.sum(torch.abs(R))
        loss = simseLoss + gmLoss
        return loss


class ContextLoss(nn.Module):
    """
    Masked L1 loss on non-water
    """

    def __call__(self, input, target, mask):
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))


class ReconstructionLoss(nn.Module):
    """
    Masked L1 loss on water
    """

    def __call__(self, input, target, mask):
        return torch.mean(torch.abs(torch.mul((input - target), mask)))


##################################################################################
# VGG network definition
##################################################################################

# Source: https://github.com/NVIDIA/pix2pixHD
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Source: https://github.com/NVIDIA/pix2pixHD
class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = Vgg19().to(device).eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def get_losses(opts, verbose, device=None):
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

    losses = {
        "G": {"a": {}, "p": {}, "tasks": {}},
        "D": {"default": {}, "advent": {}},
        "C": {},
    }

    # ------------------------------
    # -----  Generator Losses  -----
    # ------------------------------

    # painter losses
    if "p" in opts.tasks:
        losses["G"]["p"]["gan"] = (
            HingeLoss()
            if opts.gen.p.loss == "hinge"
            else GANLoss(
                use_lsgan=False,
                soft_shift=opts.dis.soft_shift,
                flip_prob=opts.dis.flip_prob,
            )
        )
        losses["G"]["p"]["dm"] = MSELoss()
        losses["G"]["p"]["vgg"] = VGGLoss(device)
        losses["G"]["p"]["tv"] = TVLoss()
        losses["G"]["p"]["context"] = ContextLoss()
        losses["G"]["p"]["reconstruction"] = ReconstructionLoss()
        losses["G"]["p"]["featmatch"] = FeatMatchLoss()

    # depth losses
    if "d" in opts.tasks:
        if not opts.gen.d.classify.enable:
            if opts.gen.d.loss == "dada":
                depth_func = DADADepthLoss()
            else:
                depth_func = SIGMLoss(opts.train.lambdas.G.d.gml)
        else:
            depth_func = CrossEntropy()

        losses["G"]["tasks"]["d"] = depth_func

    # segmentation losses
    if "s" in opts.tasks:
        losses["G"]["tasks"]["s"] = {}
        losses["G"]["tasks"]["s"]["crossent"] = CrossEntropy()
        losses["G"]["tasks"]["s"]["minent"] = MinentLoss()
        losses["G"]["tasks"]["s"]["advent"] = ADVENTAdversarialLoss(
            opts, gan_type=opts.dis.s.gan_type
        )

    # masker losses
    if "m" in opts.tasks:
        losses["G"]["tasks"]["m"] = {}
        losses["G"]["tasks"]["m"]["bce"] = nn.BCEWithLogitsLoss()
        if opts.gen.m.use_minent_var:
            losses["G"]["tasks"]["m"]["minent"] = MinentLoss(
                version=2, lambda_var=opts.train.lambdas.advent.ent_var
            )
        else:
            losses["G"]["tasks"]["m"]["minent"] = MinentLoss()
        losses["G"]["tasks"]["m"]["tv"] = TVLoss()
        losses["G"]["tasks"]["m"]["advent"] = ADVENTAdversarialLoss(
            opts, gan_type=opts.dis.m.gan_type
        )
        losses["G"]["tasks"]["m"]["gi"] = GroundIntersectionLoss()

    # ----------------------------------
    # -----  Discriminator Losses  -----
    # ----------------------------------
    if "p" in opts.tasks:
        losses["D"]["p"] = losses["G"]["p"]["gan"]
    if "m" in opts.tasks or "s" in opts.tasks:
        losses["D"]["advent"] = ADVENTAdversarialLoss(opts)
    return losses


class GroundIntersectionLoss(nn.Module):
    """
    Penalize areas in ground seg but not in flood mask
    """

    def __call__(self, pred, pseudo_ground):
        return torch.mean(1.0 * ((pseudo_ground - pred) > 0.5))


def prob_2_entropy(prob):
    """
    convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class CustomBCELoss(nn.Module):
    """
    The first argument is a tensor and the second argument is an int.
    There is no need to take sigmoid before calling this function.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target):
        return self.loss(
            prediction,
            torch.FloatTensor(prediction.size())
            .fill_(target)
            .to(prediction.get_device()),
        )


class ADVENTAdversarialLoss(nn.Module):
    """
    The class is for calculating the advent loss.
    It is used to indirectly shrink the domain gap between sim and real

    _call_ function:
    prediction: torch.tensor with shape of [bs,c,h,w]
    target: int; domain label: 0 (sim) or 1 (real)
    discriminator: the discriminator model tells if a tensor is from sim or real

    output: the loss value of GANLoss
    """

    def __init__(self, opts, gan_type="GAN"):
        super().__init__()
        self.opts = opts
        if gan_type == "GAN":
            self.loss = CustomBCELoss()
        elif gan_type == "WGAN" or "WGAN_gp" or "WGAN_norm":
            self.loss = lambda x, y: -torch.mean(y * x + (1 - y) * (1 - x))
        else:
            raise NotImplementedError

    def __call__(self, prediction, target, discriminator, depth_preds=None):
        """
        Compute the GAN loss from the Advent Discriminator given
        normalized (softmaxed) predictions (=pixel-wise class probabilities),
        and int labels (target).

        Args:
            prediction (torch.Tensor): pixel-wise probability distribution over classes
            target (torch.Tensor): pixel wise int target labels
            discriminator (torch.nn.Module): Discriminator to get the loss

        Returns:
            torch.Tensor: float 0-D loss
        """
        d_out = prob_2_entropy(prediction)
        if depth_preds is not None:
            d_out = d_out * depth_preds
        d_out = discriminator(d_out)
        if self.opts.dis.m.architecture == "OmniDiscriminator":
            d_out = multiDiscriminatorAdapter(d_out, self.opts)
        loss_ = self.loss(d_out, target)
        return loss_


def multiDiscriminatorAdapter(d_out: list, opts: dict) -> torch.tensor:
    """
    Because the OmniDiscriminator does not directly return a tensor
    (but a list of tensor).
    Since there is no multilevel masker, the 0th tensor in the list is all we want.
    This Adapter returns the first element(tensor) of the list that OmniDiscriminator
    returns.
    """
    if (
        isinstance(d_out, list) and len(d_out) == 1
    ):  # adapt the multi-scale OmniDiscriminator
        if not opts.dis.p.get_intermediate_features:
            d_out = d_out[0][0]
        else:
            d_out = d_out[0]
    else:
        raise Exception(
            "Check the setting of OmniDiscriminator! "
            + "For now, we don't support multi-scale OmniDiscriminator."
        )
    return d_out


class HingeLoss(nn.Module):
    """
    Adapted from https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py
    for  the painter
    """

    def __init__(self, tensor=torch.FloatTensor):
        super().__init__()
        self.zero_tensor = None
        self.Tensor = tensor

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
            self.zero_tensor = self.zero_tensor.to(input.device)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            loss = -torch.mean(input)
        return loss

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                loss += loss_tensor
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class DADADepthLoss:
    """Defines the reverse Huber loss from DADA paper for depth prediction
    - Samples with larger residuals are penalized more by l2 term
    - Samples with smaller residuals are penalized more by l1 term
    From https://github.com/valeoai/DADA/blob/master/dada/utils/func.py
    """

    def loss_calc_depth(self, pred, label):
        n, c, h, w = pred.size()
        assert c == 1

        pred = pred.squeeze()
        label = label.squeeze()

        adiff = torch.abs(pred - label)
        batch_max = 0.2 * torch.max(adiff).item()
        t1_mask = adiff.le(batch_max).float()
        t2_mask = adiff.gt(batch_max).float()
        t1 = adiff * t1_mask
        t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
        t2 = t2 * t2_mask
        return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)

    def __call__(self, pred, label):
        return self.loss_calc_depth(pred, label)
