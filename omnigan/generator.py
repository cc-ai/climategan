"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
import torch.nn as nn
from omnigan.tutils import init_weights
from omnigan.blocks import SpadeDecoder, BaseDecoder
from omnigan.encoder import DeeplabEncoder, BaseEncoder
import omnigan.strings as strings
from omnigan.deeplabv2 import ClassifierModule

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------

# TODO think about how to use the classifier probs at inference


def get_gen(opts, latent_shape=None, verbose=0):
    G = OmniGenerator(opts, latent_shape, verbose)
    for model in G.decoders:
        net = G.decoders[model]
        if isinstance(net, nn.ModuleDict):
            for domain_model in net.keys():
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
    if G.encoder is not None and opts.gen.encoder.architecture != "deeplabv2":
        init_weights(
            G.encoder,
            init_type=opts.gen.encoder.init_type,
            init_gain=opts.gen.encoder.init_gain,
            verbose=verbose,
        )
    return G


class OmniGenerator(nn.Module):
    def __init__(self, opts, latent_shape=None, verbose=None):
        """Creates the generator. All decoders listed in opts.gen will be added
        to the Generator.decoders ModuleDict if opts.gen.DecoderInitial is not True.
        Then can be accessed as G.decoders.T or G.decoders["T"] for instance,
        for the image Translation decoder

        Args:
            opts (addict.Dict): configuration dict
        """
        super().__init__()
        self.opts = opts

        self.encoder = None
        if "m" in opts.tasks or "simclr" in opts.tasks:
            if opts.gen.encoder.architecture == "deeplabv2":
                self.encoder = DeeplabEncoder(opts)
            else:
                self.encoder = BaseEncoder(opts)

        self.verbose = verbose
        self.decoders = {}

        if "d" in opts.tasks and not opts.gen.d.ignore:
            self.decoders["d"] = DepthDecoder(opts)

        if "s" in opts.tasks and not opts.gen.s.ignore:
            self.decoders["s"] = SegmentationDecoder(opts)

        if "m" in opts.tasks and not opts.gen.m.ignore:
            self.decoders["m"] = MaskDecoder(opts)

        self.decoders = nn.ModuleDict(self.decoders)

        if "p" in self.opts.tasks:
            self.painter = FullSpadeGen(opts)
        else:
            self.painter = nn.Module()

    def encode(self, x):
        assert self.encoder is not None
        return self.encoder.forward(x)

    def __str__(self):
        return strings.generator(self)


class MaskDecoder_(BaseDecoder):
    def __init__(self, opts):
        super().__init__(
            n_upsample=opts.gen.m.n_upsample,
            n_res=opts.gen.m.n_res,
            input_dim=opts.gen.encoder.res_dim,
            proj_dim=opts.gen.m.proj_dim,
            output_dim=opts.gen.m.output_dim,
            res_norm=opts.gen.m.res_norm,
            activ=opts.gen.m.activ,
            pad_type=opts.gen.m.pad_type,
            output_activ="sigmoid",
        )


class MaskDecoder(nn.Module):
    def __init__(self, opts):
        super(MaskDecoder, self).__init__()
        self.multi_level = opts.dis.m.multi_level
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], 2)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], 2)

    def forward(self, x):
        layer3_out, layer4_out, _ = x
        if self.multi_level:
            x1 = self.layer5(layer3_out)  # produce segmap 1
        else:
            x1 = None
        x2 = self.layer6(layer4_out)  # produce segmap 2
        return x1, x2

    # def __str__(self):
    #     return strings.basedecoder(self)


class DepthDecoder(BaseDecoder):
    def __init__(self, opts):
        super().__init__(
            n_upsample=opts.gen.d.n_upsample,
            n_res=opts.gen.d.n_res,
            input_dim=opts.gen.encoder.res_dim,
            proj_dim=opts.gen.d.proj_dim,
            output_dim=opts.gen.d.output_dim,
            res_norm=opts.gen.d.res_norm,
            activ=opts.gen.d.activ,
            pad_type=opts.gen.d.pad_type,
            output_activ="sigmoid",
        )


class SegmentationDecoder(BaseDecoder):
    def __init__(self, opts):
        super().__init__(
            n_upsample=opts.gen.s.n_upsample,
            n_res=opts.gen.s.n_res,
            input_dim=opts.gen.encoder.res_dim,
            proj_dim=opts.gen.s.proj_dim,
            output_dim=opts.gen.s.output_dim,
            res_norm=opts.gen.s.res_norm,
            activ=opts.gen.s.activ,
            pad_type=opts.gen.s.pad_type,
            output_activ="sigmoid",
        )


class FullSpadeGen(nn.Module):
    def __init__(self, opts):
        super(FullSpadeGen, self).__init__()

        n_downsample = opts.gen.p.spade_n_up

        self.latent_dim = opts.gen.p.latent_dim
        self.batch_size = opts.data.loaders.batch_size

        # Get size of latent vector based on downsampling:
        self.dec = SpadeDecoder(
            latent_dim=self.latent_dim,
            cond_nc=3,
            spade_n_up=n_downsample,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

    def forward(self, z, cond):
        return self.dec(z, cond)
