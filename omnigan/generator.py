import torch.nn as nn
from omnigan.utils import init_weights

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------


def get_gen(opts):
    G = OmniGenerator(opts)
    init_weights(G)
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

        self.E = Encoder(opts)

        self.decoders = {}

        if "A" in opts.tasks and not opts.gen.A.ignore:
            self.decoders["A"] = nn.ModuleDict(
                {"r": AdapatationDecoder(opts), "s": AdapatationDecoder(opts)}
            )

        if "D" in opts.tasks and not opts.gen.D.ignore:
            self.decoders["D"] = DepthDecoder(opts)

        if "H" in opts.tasks and not opts.gen.H.ignore:
            self.decoders["H"] = HeightDecoder(opts)

        if "T" in opts.tasks and not opts.gen.T.ignore:
            self.decoders["T"] = nn.ModuleDict(
                {"f": TranslationDecoder(opts), "n": TranslationDecoder(opts)}
            )

        if "W" in opts.tasks and not opts.gen.W.ignore:
            self.decoders["W"] = WaterDecoder(opts)

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


class AdapatationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)
