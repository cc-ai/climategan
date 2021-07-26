"""File for all blocks which are parts of decoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import climategan.strings as strings
from climategan.norms import SPADE, AdaptiveInstanceNorm2d, LayerNorm, SpectralNorm


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
        return F.interpolate(
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
        use_dada=False,
    ):
        super().__init__()

        self.low_level_feats_dim = low_level_feats_dim
        self.use_dada = use_dada

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

    def forward(self, z, cond=None, z_depth=None):
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

        if z_depth is not None and self.use_dada:
            z = z * z_depth

        if self.proj_conv is not None:
            z = self.proj_conv(z)

        if low_level_feat is not None:
            z = self.merge_feats_conv(torch.cat([low_level_feat, z], dim=1))

        return self.model(z)

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
            raise NotImplementedError(
                "The type of activation is not supported: {}".format(
                    self.last_activation
                )
            )

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
