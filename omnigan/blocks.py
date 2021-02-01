"""File for all blocks which are parts of decoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omnigan.norms import SPADE, SpectralNorm, LayerNorm, AdaptiveInstanceNorm2d
import omnigan.strings as strings
from omnigan.utils import find_target_size

# TODO: Organise file


class InterpolateNearest2d(nn.Module):
    """
    Custom implementation of nn.Upsample because pytorch/xla
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
        stride=1,
        padding=0,
        dilation=1,
        norm="none",
        activation="relu",
        pad_type="zero",
        bias=True,
    ):
        super().__init__()
        self.use_bias = bias
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
        use_spectral_norm = False
        if norm.startswith("spectral_"):
            norm = norm.replace("spectral_", "")
            use_spectral_norm = True

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
        elif norm == "spectral" or norm.startswith("spectral_"):
            self.norm = None  # dealt with later in the code
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        # initialize convolution
        if norm == "spectral" or use_spectral_norm:
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()

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
        super().__init__()
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
        norm="batch",
        activ="relu",
        pad_type="zero",
        output_activ="tanh",
        low_level_feats_dim=-1,
    ):
        super().__init__()

        self.low_level_feats_dim = low_level_feats_dim

        self.model = []
        if proj_dim != -1:
            self.proj_conv = Conv2dBlock(
                input_dim, proj_dim, 1, 1, 0, norm=norm, activation=activ
            )
        else:
            self.proj_conv = None
            proj_dim = input_dim

        if low_level_feats_dim > 0:
            self.low_level_conv = Conv2dBlock(
                input_dim=low_level_feats_dim,
                output_dim=proj_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=pad_type,
                norm=norm,
                activation=activ,
            )
            self.merge_feats_conv = Conv2dBlock(
                input_dim=2 * proj_dim,
                output_dim=proj_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                pad_type=pad_type,
                norm=norm,
                activation=activ,
            )
        else:
            self.low_level_conv = None

        self.model += [ResBlocks(n_res, proj_dim, norm, activ, pad_type=pad_type)]
        dim = proj_dim
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                InterpolateNearest2d(scale_factor=2),
                Conv2dBlock(
                    input_dim=dim,
                    output_dim=dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_type=pad_type,
                    norm=norm,
                    activation=activ,
                ),
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                input_dim=dim,
                output_dim=output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_type=pad_type,
                norm="none",
                activation=output_activ,
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, z, cond=None):
        low_level_feat = None
        if isinstance(z, (list, tuple)):
            if self.low_level_conv is None:
                z = z[0]
            else:
                z, low_level_feat = z
                low_level_feat = self.low_level_conv(low_level_feat)
                low_level_feat = F.interpolate(
                    low_level_feat, size=z.shape[-2:], mode="bilinear"
                )

        if self.proj_conv is not None:
            z = self.proj_conv(z)

        if low_level_feat is not None:
            z = self.merge_feats_conv(torch.cat([low_level_feat, z], dim=1))

        return self.model(z)

    def __str__(self):
        return strings.basedecoder(self)


class DADADepthRegressionDecoder(nn.Module):
    """
    Depth decoder based on depth auxiliary task in DADA paper
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

        mid_dim = 512

        self.do_feat_fusion = False
        if "s" in opts.tasks and opts.gen.s.depth_feat_fusion:
            self.do_feat_fusion = True
            self.dec4 = Conv2dBlock(
                128,
                res_dim,
                1,
                stride=1,
                padding=0,
                bias=True,
                activation="lrelu",
                norm="none",
            )

        self.relu = nn.ReLU(inplace=True)
        self.enc4_1 = Conv2dBlock(
            res_dim,
            mid_dim,
            1,
            stride=1,
            padding=0,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.enc4_2 = Conv2dBlock(
            mid_dim,
            mid_dim,
            3,
            stride=1,
            padding=1,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.enc4_3 = Conv2dBlock(
            mid_dim,
            128,
            1,
            stride=1,
            padding=0,
            bias=False,
            activation="lrelu",
            pad_type="reflect",
            norm="batch",
        )
        self.upsample = None
        if opts.gen.d.upsample_featuremaps:
            self.upsample = nn.Sequential(
                *[
                    InterpolateNearest2d(),
                    Conv2dBlock(
                        128,
                        32,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                        activation="lrelu",
                        pad_type="reflect",
                        norm="batch",
                    ),
                    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                ]
            )
        self._target_size = find_target_size(opts, "d")
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

    def forward(self, z):
        if isinstance(z, (list, tuple)):
            z = z[0]
        z4_enc = self.enc4_1(z)
        z4_enc = self.enc4_2(z4_enc)
        z4_enc = self.enc4_3(z4_enc)

        z_depth = None
        if self.do_feat_fusion:
            z_depth = self.dec4(z4_enc)

        if self.upsample is not None:
            z4_enc = self.upsample(z4_enc)

        depth = torch.mean(z4_enc, dim=1, keepdim=True)  # DADA paper decoder
        if depth.shape[-1] != self._target_size:
            depth = F.interpolate(
                depth,
                size=(384, 384),  # size used in MiDaS inference
                mode="bicubic",  # what MiDaS uses
                align_corners=False,
            )

            depth = F.interpolate(
                depth, (self._target_size, self._target_size), mode="nearest"
            )  # what we used in the transforms to resize input

        return depth, z_depth

    def __str__(self):
        return "Depth_decoder"
        # return strings.basedecoder(self)


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
        last_activation=None,
    ):
        super().__init__()
        # Attributes

        self.fin = fin
        self.fout = fout
        self.use_spectral_norm = spade_use_spectral_norm
        self.param_free_norm = spade_param_free_norm
        self.kernel_size = spade_kernel_size

        self.learned_shortcut = fin != fout
        self.last_activation = last_activation
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
        if self.last_activation == "lrelu":
            return self.activation(out)
        elif self.last_activation is None:
            return out
        else:
            raise NotImplementedError("The type of activation is not supported")

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

    def set_latent_shape(self, shape, is_input=True):
        """
        Sets the latent shape to start the upsampling from, i.e. z_h and z_w.
        If is_input is True, then this is the actual input shape which should
        be divided by 2 ** spade_n_up
        Otherwise, just sets z_h and z_w from shape[-2] and shape[-1]

        Args:
            shape (tuple): The shape to start sampling from.
            is_input (bool, optional): Whether to divide shape by 2 ** spade_n_up
        """
        if isinstance(shape, (list, tuple)):
            self.z_h = shape[-2]
            self.z_w = shape[-1]
        elif isinstance(shape, int):
            self.z_h = self.z_w = shape
        else:
            raise ValueError("Unknown shape type:", shape)

        if is_input:
            self.z_h = self.z_h // (2 ** self.spade_n_up)
            self.z_w = self.z_w // (2 ** self.spade_n_up)

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
