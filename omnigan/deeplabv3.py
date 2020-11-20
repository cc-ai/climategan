"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omnigan.blocks import ASPP
from pathlib import Path


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
        self,
        block,
        layers,
        output_stride,
        BatchNorm,
        pretrained=True,
        pretrained_path=None,
        no_init=False,
    ):
        self.inplanes = 64
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

        self.pretrained_path = pretrained_path

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
        # self.layer4 = self._make_layer(
        #     block, 512, layers[3], stride=strides[3],
        #     dilation=dilations[3], BatchNorm=BatchNorm
        # )
        if not no_init:
            self._init_weight()

        if pretrained and not no_init:
            self._load_pretrained_model()

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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        assert self.pretrained_path is not None
        assert Path(self.pretrained_path).exists()

        pretrain_dict = torch.load(self.pretrained_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print("    - Loaded pre-trained ResNet101")


def ResNet101(
    output_stride=8,
    BatchNorm=nn.BatchNorm2d,
    pretrained=True,
    pretrained_path=None,
    no_init=False,
):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        no_init=no_init,
    )
    return model


def build_aspp(backbone, output_stride, BatchNorm, no_init):
    return ASPP(backbone, output_stride, BatchNorm, no_init)


"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
"""


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, no_init):
        super(Decoder, self).__init__()
        if backbone == "resnet":
            low_level_inplanes = 256
        elif backbone == "mobilenet":
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        if not no_init:
            self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, no_init):
    return Decoder(num_classes, backbone, BatchNorm, no_init)


"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/mobilenet.py
"""


def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True),
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    0,
                    dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    0,
                    dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class MobileNetV2(nn.Module):
    def __init__(
        self,
        output_stride=8,
        BatchNorm=None,
        width_mult=1.0,
        pretrained=True,
        pretrained_path=None,
        no_init=False,
    ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        self.pretrained_path = pretrained_path
        interverted_residual_setting = [
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
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            stride,
                            dilation,
                            t,
                            BatchNorm,
                        )
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, dilation, t, BatchNorm)
                    )
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.loaded_pre_trained = False

        if not no_init:
            self._initialize_weights()

        if pretrained and not no_init:
            self._load_pretrained_model()

        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def _load_pretrained_model(self):
        assert self.pretrained_path is not None
        assert Path(self.pretrained_path).exists()

        pretrain_dict = torch.load(self.pretrained_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        self.loaded_pre_trained = True
        print("    - Loaded pre-trained MobileNetV2")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


"""
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/__init__.py
"""


def build_backbone(opts, no_init):
    backbone = opts.gen.deeplabv3.backbone
    output_stride = opts.gen.deeplabv3.output_stride
    use_pretrained = opts.gen.deeplabv3.use_pretrained
    if backbone == "resnet":
        return ResNet101(
            output_stride=output_stride,
            BatchNorm=nn.BatchNorm2d,
            pretrained=use_pretrained,
            pretrained_path=opts.gen.deeplabv3.pretrained_model.resnet,
            no_init=no_init,
        )
    elif backbone == "mobilenet":
        return MobileNetV2(
            output_stride=output_stride,
            BatchNorm=nn.BatchNorm2d,
            pretrained=use_pretrained,
            pretrained_path=opts.gen.deeplabv3.pretrained_model.mobilenet,
            no_init=no_init,
        )
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

        BatchNorm = nn.BatchNorm2d
        num_classes = opts.gen.s.output_dim
        backbone = opts.gen.deeplabv3.backbone
        output_stride = opts.gen.deeplabv3.output_stride

        self.aspp = build_aspp(backbone, output_stride, BatchNorm, no_init)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, no_init)

        self.freeze_bn = freeze_bn

        self._target_size = None

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
