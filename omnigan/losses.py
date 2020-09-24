"""Define all losses. When possible, as inheriting from nn.Module
To send predictions to target.device
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from random import random as rand
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


class FeatMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterionFeat = torch.nn.L1Loss()

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
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        return self.loss(logits, target.to(logits.device).long())


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


class MiniEntLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, prediction):
        assert prediction.dim() == 4
        n, c, h, w = prediction.size()
        return -torch.sum(torch.mul(prediction, torch.log2(prediction + 1e-30))) / (
            n * h * w * np.log2(c)
        )


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def entropy_loss_v2(v, lambda_var=0.1):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    entropy_map = -torch.mul(v, torch.log2(v + 1e-30)) / np.log2(c)
    entropy_map_demean = entropy_map - torch.sum(entropy_map) / (n * h * w)
    entropy_map_squ = torch.mul(entropy_map_demean, entropy_map_demean)
    return torch.sum(entropy_map + lambda_var * entropy_map_squ) / (n * h * w)


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


class SIMSELoss(nn.Module):
    """Scale invariant MSE Loss
    """

    def __init__(self):
        super(SIMSELoss, self).__init__()

    def __call__(self, prediction, target):
        d = prediction - target
        diff = torch.mean(d * d)
        relDiff = torch.mean(d) * torch.mean(d)
        return diff - relDiff

class SIGMLoss(nn.Module):
    def __init__(self, gmweight, device='cuda'):
        super(SIGMLoss, self).__init__()
        self.gmweight = gmweight
        sobelx = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
        sobely = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]])
        self.filter = (0.5 * (sobelx + sobely)).to(device)
        self.simse = SIMSELoss()
    def __call__(self, prediction, target):
        # get gradient map with sobel filters
        batch_size = prediction.size()[0]
        self.filter = (self.filter).expand((batch_size, 1, -1, -1))
        grad_pred = F.conv2d(prediction, self.filter, stride=1)
        grad_target = F.conv2d(target, self.filter, stride=1)
        gmLoss = self.gmweight * torch.norm(grad_pred - grad_target)
        diff = self.simse(prediction, target) + gmLoss
        return diff 

class ContextLoss(nn.Module):
    """
    Masked L1 loss
    """

    def __call__(self, input, target, mask):
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))


##################################################################################
# VGG network definition
##################################################################################

# Source: https://github.com/NVIDIA/pix2pixHD
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
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
        losses["G"]["p"]["gan"] = GANLoss()
        losses["G"]["p"]["sm"] = PixelCrossEntropy()
        losses["G"]["p"]["dm"] = MSELoss()
        losses["G"]["p"]["vgg"] = VGGLoss(device)
        losses["G"]["p"]["tv"] = TVLoss(opts.train.lambdas.G.p.tv)
        losses["G"]["p"]["context"] = ContextLoss()
        losses["G"]["p"]["featmatch"] = FeatMatchLoss()

    # task losses
    # ? * add discriminator and gan loss to these task when no ground truth
    # ?   instead of noisy label
    if "d" in opts.tasks:
        losses["G"]["tasks"]["d"] = SIGMLoss(opts.train.lambdas.G.d.gml)
    if "s" in opts.tasks:
        losses["G"]["tasks"]["s"] = {}
        losses["G"]["tasks"]["s"]["crossent"] = CrossEntropy()
        losses["G"]["tasks"]["s"]["minient"] = MiniEntLoss()
        losses["G"]["tasks"]["s"]["advent"] = ADVENTAdversarialLoss(opts)
    if "m" in opts.tasks:
        losses["G"]["tasks"]["m"] = {}
        losses["G"]["tasks"]["m"]["main"] = nn.BCELoss()
        if opts.gen.m.use_minent_var:
            losses["G"]["tasks"]["m"]["minent"] = lambda x: entropy_loss_v2(
                x, lambda_var=opts.train.lambdas.advent.ent_var
            )
        else:
            losses["G"]["tasks"]["m"]["minent"] = entropy_loss
        losses["G"]["tasks"]["m"]["tv"] = TVLoss(opts.train.lambdas.G.m.tv)
        losses["G"]["tasks"]["m"]["advent"] = ADVENTAdversarialLoss(opts)

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
    losses["D"]["default"] = GANLoss(
        soft_shift=opts.dis.soft_shift, flip_prob=opts.dis.flip_prob, verbose=verbose
    )
    losses["D"]["advent"] = ADVENTAdversarialLoss(opts)
    return losses


def prob_2_entropy(prob):
    """
    convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class CustomBCELoss(nn.Module):
    """
        The first argument is a tensor and the second arguement is an int.
        There is no need to take simoid before calling this function.
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target):
        return self.loss(
            prediction,
            torch.FloatTensor(prediction.size())
            .fill_(target)
            .to(prediction.get_device()),
        )


class ADVENTAdversarialLoss(nn.Module):
    """
        TODO
    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.loss = CustomBCELoss()

    def __call__(self, prediction, target, discriminator):
        d_out = discriminator(prob_2_entropy(F.softmax(prediction, dim=1)))
        if self.opts.dis.m.architecture == "OmniDiscriminator":
            d_out = multiDiscriminatorAdapter(d_out, self.opts)
        loss_ = self.loss(d_out, target)
        return loss_


def multiDiscriminatorAdapter(d_out, opts):
    if (
        isinstance(d_out, list) and len(d_out) == 1
    ):  # adapt the multi-scale Omnidiscriminator
        if not opts.dis.p.get_intermediate_features:
            d_out = d_out[0][0]
        else:
            d_out = d_out[0]
    else:
        raise Exception(
            "Check the setting of OmniDiscriminator! For now, we don't support multi-scale Omnidiscriminator."
        )
    return d_out
