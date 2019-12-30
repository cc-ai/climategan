import torch
import torch.nn as nn
from omnigan.utils import init_weights
from omnigan.blocks import Conv2dBlock, ResBlocks, SpadeDecoder, Decoder

# * https://docs.fast.ai/vision.models.unet.html
# * Dynamic UNet FastAI

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------


def get_gen(opts, verbose=0):
    G = OmniGenerator(opts)
    for model in G.decoders:
        net = G.decoders[model]
        if isinstance(net, nn.ModuleDict):
            for domain_model in net:
                init_weights(
                    net[domain_model],
                    init_type=opts.gen[model].init_type,
                    init_gain=opts.gen[model].init_gain,
                    verbose=verbose,
                )
        else:
            init_weights(
                G.decoders[model],
                init_type=opts.gen[model].init_type,
                init_gain=opts.gen[model].init_gain,
                verbose=verbose,
            )
    return G


class Encoder(nn.Module):
    def __init__(self, opts):
        """Latent Space Encoder

        Latent space shape for image CxHxW:
        (input_dim * 2 ** n_downsample)x(H / 2 ** n_downsample)x(W / 2 ** n_downsample)

        Args:
            opts (addict.Dict): options
        """
        super().__init__()
        activ = opts.gen.encoder.activ
        dim = opts.gen.encoder.dim
        input_dim = opts.gen.encoder.input_dim
        n_downsample = opts.gen.encoder.n_downsample
        n_res = opts.gen.encoder.n_res
        norm = opts.gen.encoder.norm
        pad_type = opts.gen.encoder.pad_type

        self.model = [
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type
            )
        ]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [
                Conv2dBlock(
                    dim,
                    2 * dim,
                    4,
                    2,
                    1,
                    norm=norm,
                    activation=activ,
                    pad_type=pad_type,
                )
            ]
            dim *= 2
        # residual blocks
        self.model += [
            ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class OmniGenerator(nn.Module):
    def __init__(self, opts):
        """Creates the generator. All decoders listed in opts.gen will be added
        to the Generator.decoders ModuleDict if opts.gen.DecoderInitial is not True.
        Then can be accessed as G.decoders.T or G.decoders["T"] for instance,
        for the image Translation decoder

        Args:
            opts (addict.Dict): configuration dict
        """
        super().__init__()

        self.encoder = Encoder(opts)

        self.decoders = {}

        if "a" in opts.tasks and not opts.gen.A.ignore:
            self.decoders["a"] = nn.ModuleDict(
                {"r": AdapatationDecoder(opts), "s": AdapatationDecoder(opts)}
            )

        if "t" in opts.tasks and not opts.gen.T.ignore:
            self.decoders["t"] = nn.ModuleDict(
                {"f": TranslationDecoder(opts), "n": TranslationDecoder(opts)}
            )

        if "d" in opts.tasks and not opts.gen.D.ignore:
            self.decoders["d"] = DepthDecoder(opts)

        if "h" in opts.tasks and not opts.gen.H.ignore:
            self.decoders["h"] = HeightDecoder(opts)

        if "s" in opts.tasks and not opts.gen.H.ignore:
            self.decoders["s"] = SegmentationDecoder(opts)

        if "w" in opts.tasks and not opts.gen.W.ignore:
            self.decoders["w"] = WaterDecoder(opts)

        self.decoders = nn.ModuleDict(self.decoders)

    def forward(self, x, translate_to="f"):
        z = self.encoder(x)
        h = self.decoders["h"](z)
        d = self.decoders["d"](z)
        s = self.decoders["s"](z)
        w = self.decoders["w"](z)
        y = self.decoders["t"][translate_to](z, torch.cat([h, d, s, w], dim=1))
        return y


class HeightDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.h.n_upsample,
            opts.gen.h.n_res,
            opts.gen.h.res_dim,
            opts.gen.h.output_dim,
            res_norm=opts.gen.h.res_norm,
            activ=opts.gen.h.activ,
            pad_type=opts.gen.h.pad_type,
        )


class WaterDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.w.n_upsample,
            opts.gen.w.n_res,
            opts.gen.w.res_dim,
            opts.gen.w.output_dim,
            res_norm=opts.gen.w.res_norm,
            activ=opts.gen.w.activ,
            pad_type=opts.gen.w.pad_type,
        )


class DepthDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.d.n_upsample,
            opts.gen.d.n_res,
            opts.gen.d.res_dim,
            opts.gen.d.output_dim,
            res_norm=opts.gen.d.res_norm,
            activ=opts.gen.d.activ,
            pad_type=opts.gen.d.pad_type,
        )


class SegmentationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.s.n_upsample,
            opts.gen.s.n_res,
            opts.gen.s.res_dim,
            opts.gen.s.output_dim,
            res_norm=opts.gen.s.res_norm,
            activ=opts.gen.s.activ,
            pad_type=opts.gen.s.pad_type,
        )


class AdapatationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.a.n_upsample,
            opts.gen.a.n_res,
            opts.gen.a.res_dim,
            opts.gen.a.output_dim,
            res_norm=opts.gen.a.res_norm,
            activ=opts.gen.a.activ,
            pad_type=opts.gen.a.pad_type,
        )


class TranslationDecoder(SpadeDecoder):
    def __init__(self, opts):
        cond_nc = 0
        if "d" in opts.tasks:
            cond_nc += 1
        if "h" in opts.tasks:
            cond_nc += 1
        if "s" in opts.tasks:
            cond_nc += opts.gen.s.num_classes
        if "w" in opts.tasks:
            cond_nc += 1
        super().__init__(
            opts.gen.t.n_upsample,  # number of upsampling
            opts.gen.t.n_res,  # number of resblocks before upsampling
            opts.gen.t.res_dim,  # resblock dimension
            opts.gen.t.output_dim,  # number of channels in the output
            opts.gen.t.activ,  # activation function
            opts.gen.t.pad_type,  # padding type
            opts.gen.t.spade_use_spectral_norm,  # use spectral norm in spade blocks?
            opts.gen.t.spade_param_free_norm,  # parameter-free norm in spade blocks
            opts.gen.t.spade_kernel_size,  # 3
            cond_nc,  # number of channels in the conditioning tensor
        )
