import torch.nn as nn
from omnigan.blocks import Conv2dBlock, ResBlocks

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    inplanes,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(
        self,
        block,
        layers,
        n_res=4,
        res_norm="instance",
        activ="lrelu",
        pad_type="reflect",
        multi_level=False,
    ):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True
        )  # changed padding from 1 to 0
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # if self.multi_level:
        #     self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], 2048)
        # self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], 2048)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.layer_res = ResBlocks(
            n_res, 2048, norm=res_norm, activation=activ, pad_type=pad_type
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules["1"].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, dilation=dilation, downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def get_layer3_out(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3_out = self.layer3(x)
        return layer3_out

    def deeplabv2layer4(self, layer3_out):
        return self.layer4(layer3_out)

    def forward(self, x):
        layer4_out = self.layer4(self.get_layer3_out(x))
        x = self.layer_res(layer4_out)
        return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     layer3_out = self.layer3(x)
    #     layer4_out = self.layer4(layer3_out)
    #     x = self.layer_res(layer4_out)
    #     return layer3_out, layer4_out, x
