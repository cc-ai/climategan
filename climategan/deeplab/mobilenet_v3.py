"""
from https://github.com/LikeLy-Journey/SegmenTron/blob/
4bc605eedde7d680314f63d329277b73f83b1c5f/segmentron/modules/basic.py#L34
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from climategan.blocks import InterpolateNearest2d


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=1,
        relu_first=True,
        bias=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        depthwise = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=inplanes,
            bias=bias,
        )
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(
                OrderedDict(
                    [
                        ("relu", nn.ReLU()),
                        ("depthwise", depthwise),
                        ("bn_depth", bn_depth),
                        ("pointwise", pointwise),
                        ("bn_point", bn_point),
                    ]
                )
            )
        else:
            self.block = nn.Sequential(
                OrderedDict(
                    [
                        ("depthwise", depthwise),
                        ("bn_depth", bn_depth),
                        ("relu1", nn.ReLU(inplace=True)),
                        ("pointwise", pointwise),
                        ("bn_point", bn_point),
                        ("relu2", nn.ReLU(inplace=True)),
                    ]
                )
            )

    def forward(self, x):
        return self.block(x)


class _ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu6=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(
        self, in_channels, out_channels, stride, norm_layer=nn.BatchNorm2d, **kwargs
    ):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(
                in_channels,
                in_channels,
                3,
                stride,
                1,
                groups=in_channels,
                norm_layer=norm_layer,
            ),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(
                _ConvBNReLU(
                    in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer
                )
            )
        layers.extend(
            [
                # dw
                _ConvBNReLU(
                    inter_channels,
                    inter_channels,
                    3,
                    stride,
                    dilation,
                    dilation,
                    groups=inter_channels,
                    relu6=True,
                    norm_layer=norm_layer,
                ),
                # pw-linear
                nn.Conv2d(inter_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pretrained_path=None, no_init=False):
        super(MobileNetV2, self).__init__()
        output_stride = 16
        self.multiplier = 1.0
        if output_stride == 32:
            dilations = [1, 1]
        elif output_stride == 16:
            dilations = [1, 2]
        elif output_stride == 8:
            dilations = [2, 4]
        else:
            raise NotImplementedError
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # building first layer
        input_channels = int(32 * self.multiplier) if self.multiplier > 1.0 else 32
        # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = _ConvBNReLU(
            3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer
        )

        # building inverted residual blocks
        self.planes = input_channels
        self.block1 = self._make_layer(
            InvertedResidual,
            self.planes,
            inverted_residual_setting[0:1],
            norm_layer=norm_layer,
        )
        self.block2 = self._make_layer(
            InvertedResidual,
            self.planes,
            inverted_residual_setting[1:2],
            norm_layer=norm_layer,
        )
        self.block3 = self._make_layer(
            InvertedResidual,
            self.planes,
            inverted_residual_setting[2:3],
            norm_layer=norm_layer,
        )
        self.block4 = self._make_layer(
            InvertedResidual,
            self.planes,
            inverted_residual_setting[3:5],
            dilations[0],
            norm_layer=norm_layer,
        )
        self.block5 = self._make_layer(
            InvertedResidual,
            self.planes,
            inverted_residual_setting[5:],
            dilations[1],
            norm_layer=norm_layer,
        )
        self.last_inp_channels = self.planes

        self.up2 = InterpolateNearest2d()

        # weight initialization
        if not no_init:
            self.pretrained_path = pretrained_path
            if pretrained_path is not None:
                self._load_pretrained_model()
            else:
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
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def _make_layer(
        self,
        block,
        planes,
        inverted_residual_setting,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(
                block(planes, out_channels, stride, t, dilation, norm_layer)
            )
            planes = out_channels
            for i in range(n - 1):
                features.append(
                    block(planes, out_channels, 1, t, norm_layer=norm_layer)
                )
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.up2(self.block5(c3))

        # x = self.features(x)
        # x = self.classifier(x.view(x.size(0), x.size(1)))
        return c4, c1

    def _load_pretrained_model(self):
        assert self.pretrained_path is not None
        assert Path(self.pretrained_path).exists()

        pretrain_dict = torch.load(self.pretrained_path)
        pretrain_dict = {k.replace("encoder.", ""): v for k, v in pretrain_dict.items()}
        model_dict = {}
        state_dict = self.state_dict()
        ignored = []
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                ignored.append(k)
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        self.loaded_pre_trained = True
        print(
            "    - Loaded pre-trained MobileNetV2: ignored {}/{} keys".format(
                len(ignored), len(pretrain_dict)
            )
        )
