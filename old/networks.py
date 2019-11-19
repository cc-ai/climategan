# from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
# blob/master/models/networks.py
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# ------------------------------------------
# ----- Get final networks for D and G -----
# ------------------------------------------


def get_norm_layer(layer_name):
    if layer_name == "spectral":
        return torch.nn.utils.spectral_norm
    if layer_name == "instance":
        return nn.InstanceNorm2d
    return nn.BatchNorm2d


def get_res_gen(conf_gen):
    norm_layer = get_norm_layer(conf_gen.norm_layer)
    gen = BaseResnetGenerator(
        conf_gen.input_nc,
        conf_gen.output_nc,
        conf_gen.ngf,
        norm_layer,
        conf_gen.dropout,
        conf_gen.n_blocks,
        conf_gen.padding_type,
    )
    init_weights(gen, conf_gen.init_type, conf_gen.init_gain)
    return gen


def get_flip_res_gen(conf_gen):
    norm_layer = get_norm_layer(conf_gen.norm_layer)
    gen = ResnetFlipGenerator(
        conf_gen.input_nc,
        conf_gen.output_nc,
        conf_gen.ngf,
        norm_layer,
        conf_gen.dropout,
        conf_gen.n_blocks,
        conf_gen.padding_type,
    )
    init_weights(gen, conf_gen.init_type, conf_gen.init_gain)
    return gen


def get_dis(conf_dis):
    norm_layer = get_norm_layer(conf_dis.norm_layer)
    disc = NLayerDiscriminator(
        conf_dis.input_nc, conf_dis.ndf, conf_dis.n_layers, norm_layer
    )
    init_weights(disc, conf_dis.init_type, conf_dis.init_gain)
    return disc


# ----------------------------------------
# ------------- Define classes -----------
# ----------------------------------------


class BaseResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a
    few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        dropout=0,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers:
                                   reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(BaseResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        # input 224
        model = [
            nn.ReflectionPad2d(3),  # 3 x 230
            nn.Conv2d(
                input_nc, ngf, kernel_size=7, padding=0, bias=use_bias
            ),  # 64 x 224
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(  # 128 x 112 > 256 x 56
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetFlipGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between
    a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        dropout=0,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            dropout (bool)      -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers:
                                   reflect | replicate | zero
        """
        super(ResnetFlipGenerator, self).__init__()
        assert n_blocks >= 0, "n_blocks should be > 0"
        self.n_blocks = n_blocks
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # ----------------------
        # ----- Downsample -----
        # ----------------------
        self.downsampler = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_dim = ngf * mult
            output_dim = ngf * mult * 2
            self.downsampler += [
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(output_dim),
                nn.ReLU(True),
            ]
        self.downsampler = nn.Sequential(*self.downsampler)

        # ---------------------------------
        # ----- Encode with ResBlocks -----
        # ---------------------------------
        self.res_encoder = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            self.res_encoder += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    use_bias=use_bias,
                )
            ]
        self.res_encoder = nn.Sequential(*self.res_encoder)

        # ------------------------------------
        # ----- Decode with ResnetBlocks -----
        # ------------------------------------
        self.res_decoder = []
        for i in range(n_blocks):  # add ResNet blocks
            self.res_decoder += [
                ResnetBlock(
                    ngf * mult + 1,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    use_bias=use_bias,
                )
            ]
        self.res_decoder = nn.Sequential(*self.res_decoder)

        # --------------------
        # ----- Upsample -----
        # --------------------
        self.upsampler = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            input_dim = ngf * mult
            output_dim = int(ngf * mult / 2)
            if i == 0:
                input_dim += 1
            self.upsampler += [
                nn.ConvTranspose2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(output_dim),
                nn.ReLU(True),
            ]
        self.upsampler += [nn.ReflectionPad2d(3)]
        self.upsampler += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.upsampler += [nn.Tanh()]
        self.upsampler = nn.Sequential(*self.upsampler)

    def forward(self, input, flip_val=1):
        """Standard forward"""
        self.downsampled = self.downsampler(input)
        self.res_encoded = self.res_encoder(self.downsampled)
        self.flip = flip_val * torch.ones(
            (input.shape[0], 1, *self.res_encoded.shape[2:])
        )
        self.flip = self.flip.to(self.res_encoded.device)
        self.res_decoded = self.res_decoder(
            torch.cat([self.flip, self.res_encoded], dim=1)
        )
        return self.upsampler(self.res_decoded)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            dropout (float)     -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block
        (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if dropout > 0:
            conv_block += [nn.Dropout(dropout)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


# ----------------------------------------
# ----------- Define utilities -----------
# ----------------------------------------


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)     -- network to be initialized
        init_type (str)   -- the name of an initialization method:
                             normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper.
    But xavier and kaiming might work better for some applications.
    Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)
