import torch
import torch.nn as nn
import torch.nn.functional as F
from climategan.blocks import InterpolateNearest2d
from climategan.utils import find_target_size


class _ASPPModule(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    def __init__(
        self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, no_init
    ):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        if not no_init:
            self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    def __init__(self, backbone, output_stride, BatchNorm, no_init):
        super().__init__()

        if backbone == "mobilenet":
            inplanes = 320
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            inplanes,
            256,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if not no_init:
            self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV2Decoder(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py
    def __init__(self, opts, no_init=False):
        super().__init__()
        self.aspp = ASPP("resnet", 16, nn.BatchNorm2d, no_init)
        self.use_dada = ("d" in opts.tasks) and opts.gen.s.use_dada

        conv_modules = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        ]
        if opts.gen.s.upsample_featuremaps:
            conv_modules = [InterpolateNearest2d(scale_factor=2)] + conv_modules

        conv_modules += [
            nn.Conv2d(256, opts.gen.s.output_dim, kernel_size=1, stride=1),
        ]
        self.conv = nn.Sequential(*conv_modules)

        self._target_size = find_target_size(opts, "s")
        print(
            "      - {}:  setting target size to {}".format(
                self.__class__.__name__, self._target_size
            )
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
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate logits to the target seg map's size"
            raise Exception(error)
        if isinstance(z, (list, tuple)):
            z = z[0]
        if z.shape[1] != 2048:
            raise Exception(
                "Segmentation decoder will only work with 2048 channels for z"
            )

        if z_depth is not None and self.use_dada:
            z = z * z_depth

        y = self.aspp(z)
        y = self.conv(y)
        return F.interpolate(y, self._target_size, mode="bilinear", align_corners=True)
