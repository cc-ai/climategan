from torch import nn
import numpy as np
from omnigan.utils import init_weights


def get_classifier(opts, latent_shape, verbose):
    latent_size = np.prod(latent_shape)
    C = OmniClassifier(opts, latent_size)
    init_weights(
        C,
        init_type=opts.classifier.init_type,
        init_gain=opts.classifier.init_gain,
        verbose=verbose,
    )
    return C


class OmniClassifier(nn.Module):
    def __init__(self, dim):
        super(OmniClassifier, self).__init__()

        self.max_pool1 = nn.MaxPool2d(2)
        self.BasicBlock1 = BasicBlock(256, 128, True)
        self.max_pool2 = nn.MaxPool2d(2)
        self.BasicBlock2 = BasicBlock(128, 64, True)
        self.avg_pool = nn.AvgPool2d((16, 16))
        self.fc = nn.Linear(64, 2)
        self.output_dim = dim

    def forward(self, x):
        max_pooled1 = self.max_pool1(x)
        res_block1 = self.BasicBlock1(max_pooled1)
        max_pooled2 = self.max_pool2(res_block1)
        res_block2 = self.BasicBlock2(max_pooled2)
        avg_pool = self.avg_pool(res_block2)
        fc_output = self.fc(avg_pool.squeeze())
        # logits = nn.functional.softmax(fc_output)
        # return logits
        return fc_output


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
        #Both conv1 and downsample layers downsample the input when stride != 1
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


def conv_block(in_channels, out_channels):
    """returns a block Convolution - batch normalization - ReLU - Pooling

    Arguments:
        in_channels {int} -- Number of channels in the input image
        out_channels {int} -- Number of channels produced by the convolution
    
    Returns:
        block -- Convolution - batch normalization - ReLU - Pooling
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
    Arguments:
        in_planes {int} -- Number of channels in the input image
        out_planes {int} -- Number of channels produced by the convolution
    
    Keyword Arguments:
        stride {int or tuple, optional} -- Stride of the convolution.
        Default: 1 (default: {1})
        groups {int, optional} -- Number of blocked connections 
        from input channels to output channels.tion] (default: {1})
        dilation {int or tuple, optional} -- Spacing between kernel elements (default: {1})
    
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
        stride {int or tuple, optional} -- Stride of the convolution. Default: 1 (default: {1})
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)