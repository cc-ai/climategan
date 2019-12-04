import torch.nn as nn
from omnigan.utils import init_weights

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
        super().__init__()
        self.project = nn.Conv2d(3, 64, 1)
        self.downsample = nn.AdaptiveMaxPool2d(32)

    def forward(self, x):
        return self.project(self.downsample(x))


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


class Decoder(nn.Module):
    """generic class for decoders
    """

    def __init__(self, opts):
        super().__init__()

    def forward(self, x):
        return self.layers(x)


class HeightDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class WaterDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class DepthDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class TranslationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class SegmentationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class AdapatationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)
