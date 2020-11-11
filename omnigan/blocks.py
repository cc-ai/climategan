"""File for all blocks which are parts of decoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omnigan.norms import SPADE, SpectralNorm, LayerNorm, AdaptiveInstanceNorm2d
import omnigan.strings as strings

# TODO: Organise file


class InterpolateNearest2d(nn.Module):
    """
    Custom implementation of nn.Upsample because pytroch/xla
    does not yet support scale_factor and needs to be provided with
    the output_size
    """

    def __init__(self, scale_factor=2):
        """
        Create an InterpolateNearest2d module

        Args:
            scale_factor (int, optional): Output size multiplier. Defaults to 2.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Interpolate x in "nearest" mode on its last 2 dimensions

        Args:
            x (torch.Tensor): input to interpolate

        Returns:
            torch.Tensor: upsampled tensor with shape
                (...x.shape, x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)
        """
        return nn.functional.interpolate(
            x,
            size=(x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor),
            mode="nearest",
        )


# -----------------------------------------
# -----  Generic Convolutional Block  -----
# -----------------------------------------
class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        norm="none",
        activation="relu",
        pad_type="zero",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "instance":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "layer":
            self.norm = LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "spectral":
            self.norm = None  # dealt with later in the code
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        # initialize convolution
        if norm == "spectral":
            self.conv = SpectralNorm(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size,
                    stride,
                    dilation=dilation,
                    bias=self.use_bias,
                )
            )
        else:
            self.conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                dilation=dilation,
                bias=self.use_bias if norm != "batch" else False,
            )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __str__(self):
        return strings.conv2dblock(self)


# -----------------------------
# -----  Residual Blocks  -----
# -----------------------------
class ResBlocks(nn.Module):
    """
    From https://github.com/NVlabs/MUNIT/blob/master/networks.py
    """

    def __init__(self, num_blocks, dim, norm="in", activation="relu", pad_type="zero"):
        super().__init__()
        self.model = nn.Sequential(
            *[
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return strings.resblocks(self)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.norm = norm
        self.activation = activation
        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

    def __str__(self):
        return strings.resblock(self)


_BOTTLENECK_EXPANSION = 4


class Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = Conv2dBlock(in_ch, mid_ch, 1, stride, 0, norm="batch")
        self.conv3x3 = Conv2dBlock(
            mid_ch, mid_ch, 3, 1, dilation, dilation, norm="batch"
        )
        self.increase = Conv2dBlock(
            mid_ch, out_ch, 1, 1, 0, 1, activation="none", norm="batch"
        )
        self.shortcut = (
            Conv2dBlock(in_ch, out_ch, 1, stride, 0, 1, activation="none", norm="batch")
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

    def __str__(self):
        return strings.bottleneck(self)


class ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(Stem, self).__init__()
        self.add_module("conv1", Conv2dBlock(3, out_ch, 7, 2, 3, 1, norm="batch"))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


# --------------------------
# -----  Base Decoder  -----
# --------------------------
class BaseDecoder(nn.Module):
    def __init__(
        self,
        n_upsample=4,
        n_res=4,
        input_dim=2048,
        proj_dim=64,
        output_dim=3,
        res_norm="instance",
        activ="relu",
        pad_type="zero",
        output_activ="tanh",
        conv_norm="layer",
    ):
        super().__init__()

        if proj_dim != -1:
            conv = Conv2dBlock(
                input_dim, proj_dim, 1, 1, 0, norm=conv_norm, activation=activ
            )
            if res_norm == "spectral":
                conv = SpectralNorm(conv)
            self.model = [conv]
        else:
            self.model = []
            proj_dim = input_dim

        self.model += [ResBlocks(n_res, proj_dim, res_norm, activ, pad_type=pad_type)]
        dim = proj_dim
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                InterpolateNearest2d(scale_factor=2),
                Conv2dBlock(
                    dim,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm=conv_norm,
                    activation=activ,
                    pad_type=pad_type,
                ),
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                dim,
                output_dim,
                7,
                1,
                3,
                norm=conv_norm,
                activation=output_activ,
                pad_type=pad_type,
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, z):
        if isinstance(z, (list, tuple)):
            z = z[0]
        return self.model(z)

    def __str__(self):
        return strings.basedecoder(self)


class DepthDecoder(nn.Module):
    """#Depth decoder based on depth auxiliary task in DADA paper

    """

    def __init__(self, opts):
        super().__init__()
        if (
            opts.gen.encoder.architecture == "deeplabv3"
            and opts.gen.deeplabv3.backbone == "mobilenet"
        ):
            res_dim = 320
        else:
            res_dim = 2048

        # if res_dim == 2048:
        #     mid_dim = 512
        # else:
        #     mid_dim = 256

        mid_dim = 512

        self.relu = nn.ReLU(inplace=True)
        self.enc4_1 = nn.Conv2d(
            res_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.enc4_2 = nn.Conv2d(
            mid_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.enc4_3 = nn.Conv2d(
            mid_dim, 128, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.output_size = opts.data.transforms[-1].new_size

    def forward(self, z):
        if isinstance(z, (list, tuple)):
            z = z[0]
        z4_enc = self.enc4_1(z)
        z4_enc = self.relu(z4_enc)
        z4_enc = self.enc4_2(z4_enc)
        z4_enc = self.relu(z4_enc)
        z4_enc = self.enc4_3(z4_enc)

        depth = torch.mean(z4_enc, dim=1, keepdim=True)  # DADA paper decoder
        depth = F.interpolate(
            depth,
            size=(384, 384),  # size used in MiDaS inference
            mode="bicubic",  # what MiDaS uses
            align_corners=False,
        )
        depth = F.interpolate(
            depth, (self.output_size, self.output_size), mode="nearest"
        )  # what we used in the transforms to resize input
        return depth

    def __str__(self):
        return strings.basedecoder(self)


# --------------------------
# -----  SPADE Blocks  -----
# --------------------------
# https://github.com/NVlabs/SPADE/blob/0ff661e70131c9b85091d11a66e019c0f2062d4c
# /models/networks/generator.py
# 0ff661e on 13 Apr 2019
class SPADEResnetBlock(nn.Module):
    def __init__(
        self,
        fin,
        fout,
        cond_nc,
        spade_use_spectral_norm,
        spade_param_free_norm,
        spade_kernel_size,
    ):
        super().__init__()
        # Attributes

        self.fin = fin
        self.fout = fout
        self.use_spectral_norm = spade_use_spectral_norm
        self.param_free_norm = spade_param_free_norm
        self.kernel_size = spade_kernel_size

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spade_use_spectral_norm:
            self.conv_0 = SpectralNorm(self.conv_0)
            self.conv_1 = SpectralNorm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = SpectralNorm(self.conv_s)

        self.norm_0 = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)
        self.norm_1 = SPADE(spade_param_free_norm, spade_kernel_size, fmiddle, cond_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.activation(self.norm_0(x, seg)))
        dx = self.conv_1(self.activation(self.norm_1(dx, seg)))

        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def activation(self, x):
        return F.leaky_relu(x, 2e-1)

    def __str__(self):
        return strings.spaderesblock(self)


class PainterSpadeDecoder(nn.Module):
    def __init__(self, opts):
        """Create a SPADE-based decoder, which forwards z and the conditioning
        tensors seg (in the original paper, conditioning is on a semantic map only).
        All along, z is conditioned on seg. First 3 SpadeResblocks (SRB) do not shrink
        the channel dimension, and an upsampling is applied after each. Therefore
        2 upsamplings at this point. Then, for each remaining upsamplings
        (w.r.t. spade_n_up), the SRB shrinks channels by 2. Before final conv to get 3
        channels, the number of channels is therefore:
            final_nc = channels(z) * 2 ** (spade_n_up - 2)
        Args:
            latent_dim (tuple): z's shape (only the number of channels matters)
            cond_nc (int): conditioning tensor's expected number of channels
            spade_n_up (int): Number of total upsamplings from z
            spade_use_spectral_norm (bool): use spectral normalization?
            spade_param_free_norm (str): norm to use before SPADE de-normalization
            spade_kernel_size (int): SPADE conv layers' kernel size
        Returns:
            [type]: [description]
        """
        super().__init__()

        latent_dim = opts.gen.p.latent_dim
        cond_nc = 3
        spade_n_up = opts.gen.p.spade_n_up
        spade_use_spectral_norm = opts.gen.p.spade_use_spectral_norm
        spade_param_free_norm = opts.gen.p.spade_param_free_norm
        spade_kernel_size = 3

        self.z_nc = latent_dim
        self.spade_n_up = spade_n_up

        self.fc = nn.Conv2d(3, latent_dim, 3, padding=1)
        self.head_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.G_middle_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.G_middle_1 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.up_spades = nn.Sequential(
            *[
                SPADEResnetBlock(
                    self.z_nc // 2 ** i,
                    self.z_nc // 2 ** (i + 1),
                    cond_nc,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                )
                for i in range(spade_n_up - 2)
            ]
        )

        self.final_nc = self.z_nc // 2 ** (spade_n_up - 2)

        self.final_spade = SPADEResnetBlock(
            self.final_nc,
            self.final_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.final_shortcut = None
        if opts.gen.p.use_final_shortcut:
            self.final_shortcut = nn.Sequential(
                *[
                    SpectralNorm(nn.Conv2d(self.final_nc, 3, 1)),
                    nn.BatchNorm2d(3),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        self.conv_img = nn.Conv2d(self.final_nc, 3, 3, padding=1)

        self.upsample = InterpolateNearest2d(scale_factor=2)

    def _apply(self, fn):
        # print("Applying SpadeDecoder", fn)
        super()._apply(fn)
        # self.head_0 = fn(self.head_0)
        # self.G_middle_0 = fn(self.G_middle_0)
        # self.G_middle_1 = fn(self.G_middle_1)
        # for i, up in enumerate(self.up_spades):
        #     self.up_spades[i] = fn(up)
        # self.conv_img = fn(self.conv_img)
        return self

    def forward(self, z, cond):
        if z is None:
            assert self.z_h is not None and self.z_w is not None
            z = self.fc(F.interpolate(cond, size=(self.z_h, self.z_w)))
        y = self.head_0(z, cond)
        y = self.upsample(y)
        y = self.G_middle_0(y, cond)
        y = self.upsample(y)
        y = self.G_middle_1(y, cond)

        for i, up in enumerate(self.up_spades):
            y = self.upsample(y)
            y = up(y, cond)

        if self.final_shortcut is not None:
            cond = self.final_shortcut(y)
        y = self.final_spade(y, cond)
        y = self.conv_img(F.leaky_relu(y, 2e-1))
        y = torch.tanh(y)
        return y

    def __str__(self):
        return strings.spadedecoder(self)


class _ASPPModule(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    def __init__(
        self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, no_init
    ):
        super(_ASPPModule, self).__init__()
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
