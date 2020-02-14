import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from omnigan.norms import SPADE, SpectralNorm, LayerNorm, AdaptiveInstanceNorm2d

# TODO: Organise file


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
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super().__init__()
        self.use_bias = True
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
            self.norm = spectral_norm
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

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
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == "spectral":
            self.conv = SpectralNorm(
                nn.Conv2d(
                    input_dim, output_dim, kernel_size, stride, bias=self.use_bias
                )
            )
        else:
            self.conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride, bias=self.use_bias
            )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# -----------------------------
# -----  Residual Blocks  -----
# -----------------------------
class ResBlocks(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

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


# --------------------------
# -----  Base Decoder  -----
# --------------------------
class BaseDecoder(nn.Module):
    def __init__(
        self,
        n_upsample=4,
        n_res=4,
        dim=64,
        output_dim=3,
        res_norm="instance",
        activ="relu",
        pad_type="zero",
    ):
        super().__init__()

        self.model = [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(
                    dim,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm="layer",
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
                norm="none",
                activation="tanh",
                pad_type=pad_type,
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


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

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spade_use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

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


class SpadeDecoder(BaseDecoder):
    def __init__(
        self,
        latent_shape,
        num_upsampling_layers,
        spade_use_spectral_norm,
        spade_param_free_norm,
        spade_kernel_size,
        cond_nc,
    ):
        super().__init__()

        self.zdim, self.sw, self.sh = latent_shape
        self.num_upsampling_layers = num_upsampling_layers

        # self.fc = nn.Conv2d(cond_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(
            self.zdim,
            self.zdim,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.G_middle_0 = SPADEResnetBlock(
            self.zdim,
            self.zdim,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.G_middle_1 = SPADEResnetBlock(
            self.zdim,
            self.zdim,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.up_spades = nn.Sequential(
            *[
                SPADEResnetBlock(
                    self.zdim // 2 ** i,
                    self.zdim // 2 ** (i + 1),
                    cond_nc,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                )
                for i in range(num_upsampling_layers)
            ]
        )

        self.final_nc = self.zdim // 2 ** num_upsampling_layers

        # self.up_0 = SPADEResnetBlock(
        #     16 * nf,
        #     8 * nf,
        #     cond_nc,
        #     spade_use_spectral_norm,
        #     spade_param_free_norm,
        #     spade_kernel_size,
        # )
        # self.up_1 = SPADEResnetBlock(
        #     8 * nf,
        #     4 * nf,
        #     cond_nc,
        #     spade_use_spectral_norm,
        #     spade_param_free_norm,
        #     spade_kernel_size,
        # )
        # self.up_2 = SPADEResnetBlock(
        #     4 * nf,
        #     2 * nf,
        #     cond_nc,
        #     spade_use_spectral_norm,
        #     spade_param_free_norm,
        #     spade_kernel_size,
        # )
        # self.up_3 = SPADEResnetBlock(
        #     2 * nf,
        #     1 * nf,
        #     cond_nc,
        #     spade_use_spectral_norm,
        #     spade_param_free_norm,
        #     spade_kernel_size,
        # )

        # final_nc = nf

        # if self.num_upsampling_layers == "most":
        #     self.up_4 = SPADEResnetBlock(
        #         1 * nf,
        #         nf // 2,
        #         cond_nc,
        #         spade_use_spectral_norm,
        #         spade_param_free_norm,
        #         spade_kernel_size,
        #     )
        #     final_nc = nf // 2

        self.conv_img = nn.Conv2d(self.final_nc, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)

    def _forward(self, z, seg):

        # TODO parameter for number of spades resblocks

        # x = F.interpolate(seg, size=(self.sh, self.sw))
        # x = self.fc(x)

        x = self.head_0(z, seg)

        x = self.upsample(x)
        x = self.G_middle_0(x, seg)

        # if self.num_upsampling_layers == "more" or self.num_upsampling_layers == "most":
        #     x = self.upsample(x)

        x = self.G_middle_1(x, seg)
        for i, up in enumerate(self.up_spades):
            print(f"Up {i}")
            x = self.upsample(x)
            x = up(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
