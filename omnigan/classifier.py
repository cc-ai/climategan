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


class domainClassifier(nn.Module):
    def __init__(self, dim):
        super(domainClassifier, self).__init__()

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
        #logits = nn.functional.softmax(fc_output)
        #return logits
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
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
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