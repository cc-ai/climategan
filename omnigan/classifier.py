"""Classifier architecture for domain adaptation
"""
from torch import nn
from omnigan.tutils import init_weights
from omnigan.blocks import Conv2dBlock


def get_classifier(opts, latent_space, verbose, no_init=False):
    C = OmniClassifier(latent_space, opts.classifier.proj_dim, opts.classifier.loss)

    if no_init:
        return C

    init_weights(
        C,
        init_type=opts.classifier.init_type,
        init_gain=opts.classifier.init_gain,
        verbose=verbose,
    )
    return C


class OmniClassifier(nn.Module):
    def __init__(self, latent_space, proj_dim, loss):
        super(OmniClassifier, self).__init__()
        assert len(latent_space) == 3
        self.channels = latent_space[0]
        self.feature_size = latent_space[1]
        self.loss = loss
        self.model = nn.Sequential(
            *[
                Conv2dBlock(self.channels, proj_dim, 3, 1, 1),
                nn.MaxPool2d(2),
                BasicBlock(proj_dim, int(proj_dim / 2), True),
                nn.MaxPool2d(2),
                BasicBlock(int(proj_dim / 2), int(proj_dim / 4), True),
                nn.AvgPool2d((int(self.feature_size / 4), int(self.feature_size / 4))),
                Squeeze(-1),
                Squeeze(-1),
                nn.Linear(int(proj_dim / 4), 2),
            ]
        )

    def forward(self, x):
        fc_output = self.model(x)
        if (self.loss == "l1") or (self.loss == "l2"):
            fc_output = nn.functional.softmax(fc_output, dim=1)

        return fc_output


class Squeeze(nn.Module):
    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim > -2:
            return x.squeeze(self.dim)
        return x.squeeze()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both conv1 and downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = downsample
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), norm_layer(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
    Arguments:
        in_planes {int} -- Number of channels in the input image
        out_planes {int} -- Number of channels produced by the convolution

    Keyword Arguments:
        stride {int or tuple, optional} -- Stride of the convolution.
        Default: 1 (default: {1})
        groups {int, optional} -- Number of blocked connections
        from input channels to output channels.tion]
        (default: {1}) dilation {int or tuple, optional} --
        Spacing between kernel elements (default: {1})

    Returns:
        output layer of 3x3 convolution with padding
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    Arguments:
        in_planes {int} -- Number of channels in the input image
        out_planes {int} -- Number of channels produced by the convolution

    Keyword Arguments:
        stride {int or tuple, optional} -- Stride of the convolution.
        Default: 1 (default: {1})
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
