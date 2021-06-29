"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from climategan.deeplab.mobilenet_v3 import SeparableConv2d
from climategan.utils import find_target_size


class _DeepLabHead(nn.Module):
    def __init__(
        self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        last_channels = c4_channels
        # self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        # last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(
                last_channels, 256, 3, norm_layer=norm_layer, relu_first=False
            ),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1),
        )

    def forward(self, x, c1=None):
        return self.block(x)


class ConvBNReLU(nn.Module):
    """
    https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes/blob/master/models/deeplabv3plus.py
    """

    def __init__(
        self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ASPPv3Plus(nn.Module):
    """
    https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes/blob/master/models/deeplabv3plus.py
    """

    def __init__(self, backbone, no_init):
        super().__init__()

        if backbone == "mobilenet":
            in_chan = 320
        else:
            in_chan = 2048

        self.with_gp = False
        self.conv1 = ConvBNReLU(in_chan, 256, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 256, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, 256, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, 256, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, 256, ks=1)
            self.conv_out = ConvBNReLU(256 * 5, 256, ks=1)
        else:
            self.conv_out = ConvBNReLU(256 * 4, 256, ks=1)

        if not no_init:
            self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode="bilinear", align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


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


"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py
"""


class DeepLabV3Decoder(nn.Module):
    def __init__(
        self,
        opts,
        no_init=False,
        freeze_bn=False,
    ):
        super().__init__()

        num_classes = opts.gen.s.output_dim
        self.backbone = opts.gen.deeplabv3.backbone
        self.use_dada = ("d" in opts.tasks) and opts.gen.s.use_dada

        if self.backbone == "resnet":
            self.aspp = ASPPv3Plus(self.backbone, no_init)
            self.decoder = Decoder(num_classes)

            self.freeze_bn = freeze_bn
        else:
            self.head = _DeepLabHead(num_classes, c4_channels=320)

        self._target_size = find_target_size(opts, "s")
        print(
            "      - {}:  setting target size to {}".format(
                self.__class__.__name__, self._target_size
            )
        )

        if not no_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

            self.load_pretrained(opts)

    def load_pretrained(self, opts):
        assert opts.gen.deeplabv3.backbone in {"resnet", "mobilenet"}
        assert Path(opts.gen.deeplabv3.pretrained_model.resnet).exists()
        if opts.gen.deeplabv3.backbone == "resnet":
            std = torch.load(opts.gen.deeplabv3.pretrained_model.resnet)
            self.aspp.load_state_dict(
                {
                    k.replace("aspp.", ""): v
                    for k, v in std.items()
                    if k.startswith("aspp.")
                }
            )
            self.decoder.load_state_dict(
                {
                    k.replace("decoder.", ""): v
                    for k, v in std.items()
                    if k.startswith("decoder.")
                    and not (len(v.shape) > 0 and v.shape[0] == 19)
                },
                strict=False,
            )
            print(
                "- Loaded pre-trained DeepLabv3+ (Resnet) Decoder & ASPP as Seg Decoder"
            )
        else:
            std = torch.load(opts.gen.deeplabv3.pretrained_model.mobilenet)
            self.load_state_dict(
                {
                    k: v
                    for k, v in std.items()
                    if k.startswith("head.")
                    and not (len(v.shape) > 0 and v.shape[0] == 19)
                },
                strict=False,
            )
            print(
                "    - Loaded pre-trained DeepLabv3+ (MobileNetV2) Head as Seg Decoder"
            )

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

    def forward(self, z, z_depth=None):
        assert isinstance(z, (tuple, list))
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate logits to the target seg map's size"
            raise ValueError(error)

        z_high, z_low = z

        if z_depth is not None and self.use_dada:
            z_high = z_high * z_depth

        if self.backbone == "resnet":
            z_high = self.aspp(z_high)
            s = self.decoder(z_high, z_low)
        else:
            s = self.head(z_high)

        s = F.interpolate(
            s, size=self._target_size, mode="bilinear", align_corners=True
        )

        return s

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
