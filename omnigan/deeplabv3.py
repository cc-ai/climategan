"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnigan.blocks import ASPPv3Plus, ConvBNReLU


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, output_stride, BatchNorm, verbose=0, no_init=False
    ):
        self.inplanes = 64
        self.verbose = verbose
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.layer4 = self._make_MG_unit(
            block,
            512,
            blocks=blocks,
            stride=strides[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(
        self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat


def ResNet101(output_stride=8, BatchNorm=nn.BatchNorm2d, verbose=0, no_init=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        output_stride,
        BatchNorm,
        verbose=verbose,
        no_init=no_init,
    )
    return model


class Decoder(nn.Module):
    """
    https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes/blob/master/models/deeplabv3plus.py
    """

    def __init__(self, n_classes):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(256, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
            ConvBNReLU(304, 256, ks=3, padding=1),
            ConvBNReLU(256, 256, ks=3, padding=1),
        )
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(
            feat_aspp, (H, W), mode="bilinear", align_corners=True
        )
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        logits = self.conv_out(feat_out)
        return logits


def build_backbone(opts, no_init, verbose=0):
    backbone = opts.gen.deeplabv3.backbone
    output_stride = opts.gen.deeplabv3.output_stride
    if backbone == "resnet":
        resnet = ResNet101(
            output_stride=output_stride,
            BatchNorm=nn.BatchNorm2d,
            verbose=verbose,
            no_init=no_init,
        )
        if not no_init:
            assert opts.gen.deeplabv3.backbone == "resnet"
            assert Path(opts.gen.deeplabv3.pretrained_model.resnet).exists()

            std = torch.load(opts.gen.deeplabv3.pretrained_model.resnet)
            resnet.load_state_dict(
                {
                    k.replace("backbone.", ""): v
                    for k, v in std.items()
                    if k.startswith("backbone.")
                }
            )
            print("- Loaded pre-trained DeepLabv3+ Backbone as Encoder")
        return resnet
    else:
        raise NotImplementedError("Unknown backbone in " + str(opts.gen.deeplabv3))


"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py
"""


class DeepLabV3Decoder(nn.Module):
    def __init__(
        self, opts, no_init=False, freeze_bn=False,
    ):
        super().__init__()

        num_classes = opts.gen.s.output_dim
        backbone = opts.gen.deeplabv3.backbone

        self.aspp = ASPPv3Plus(backbone, no_init)
        self.decoder = Decoder(num_classes)

        self.freeze_bn = freeze_bn

        self._target_size = None

        if not no_init:
            self.load_pretrained(opts)

    def load_pretrained(self, opts):
        assert opts.gen.deeplabv3.backbone == "resnet"
        assert Path(opts.gen.deeplabv3.pretrained_model.resnet).exists()

        std = torch.load(opts.gen.deeplabv3.pretrained_model.resnet)
        self.aspp.load_state_dict(
            {k.replace("aspp.", ""): v for k, v in std.items() if k.startswith("aspp.")}
        )
        self.decoder.load_state_dict(
            {
                k.replace("decoder.", ""): v
                for k, v in std.items()
                if k.startswith("decoder.")
                and not (len(v.shape) > 2 and v.shape[0] == 19)
            },
            strict=False,
        )
        print("- Loaded pre-trained DeepLabv3+ Decoder & ASPP as Segmentation Decoder")

    def set_target_size(self, size):
        """
        Set final interpolation's target size

        Args:
            size (int, list, tuple): target size (h, w). If int, target will be (i, i)
        """
        if isinstance(size, (list, tuple)):
            self._target_size = size[:2]
        else:
            self._target_size = (size, size)

    def forward(self, z):
        assert isinstance(z, (tuple, list))
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate logits to the target seg map's size"
            raise ValueError(error)

        x, low_level_feat = z
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(
            x, size=self._target_size, mode="bilinear", align_corners=True
        )

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
