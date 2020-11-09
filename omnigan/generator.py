"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
import torch
import torch.nn as nn
from omnigan.tutils import init_weights
from omnigan.blocks import (
    PainterSpadeDecoder,
    BaseDecoder,
    ASPP,
    InterpolateNearest2d,
)
from omnigan.encoder import DeeplabEncoder, BaseEncoder
import omnigan.strings as strings
import torch.nn.functional as F

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----  latent space and decode to (3 or 1) x 256 x 256               -----
# --------------------------------------------------------------------------


def get_gen(opts, latent_shape=None, verbose=0, no_init=False):
    G = OmniGenerator(opts, latent_shape, verbose, no_init)
    if no_init:
        return G

    for model in G.decoders:
        net = G.decoders[model]
        if isinstance(net, nn.ModuleDict):
            for domain, domain_model in net.items():
                init_weights(
                    net[domain_model],
                    init_type=opts.gen[model].init_type,
                    init_gain=opts.gen[model].init_gain,
                    verbose=verbose,
                    caller=f"get_gen decoder {model} {domain}"
                )
        else:
            init_weights(
                G.decoders[model],
                init_type=opts.gen[model].init_type,
                init_gain=opts.gen[model].init_gain,
                verbose=verbose,
                caller=f"get_gen decoder {model}"
            )
    if G.encoder is not None and opts.gen.encoder.architecture != "deeplabv2":
        init_weights(
            G.encoder,
            init_type=opts.gen.encoder.init_type,
            init_gain=opts.gen.encoder.init_gain,
            verbose=verbose,
            caller=f"get_gen encoder"
        )
    # Init painter weights
    init_weights(
        G.painter,
        init_type=opts.gen.p.init_type,
        init_gain=opts.gen.p.init_gain,
        verbose=verbose,
        caller=f"get_gen painter"
    )
    return G


class OmniGenerator(nn.Module):
    def __init__(self, opts, latent_shape=None, verbose=None, no_init=False):
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
        if "m" in opts.tasks:
            if opts.gen.encoder.architecture == "deeplabv2":
                self.encoder = DeeplabEncoder(opts, no_init)
                print("  - Created Deeplab Encoder")
            else:
                self.encoder = BaseEncoder(opts)
                print("  - Created Base Encoder")

        self.verbose = verbose
        self.decoders = {}

        if "d" in opts.tasks and not opts.gen.d.ignore:
            self.decoders["d"] = DepthDecoder(opts)
            print("  - Created Depth Decoder")

        if "s" in opts.tasks and not opts.gen.s.ignore:
            print("  - Created Segmentation Decoder")
            self.decoders["s"] = SegmentationDecoder(opts)

        if "m" in opts.tasks and not opts.gen.m.ignore:
            print("  - Created Mask Decoder")
            self.decoders["m"] = MaskDecoder(opts)

        self.decoders = nn.ModuleDict(self.decoders)

        if "p" in self.opts.tasks:
            self.painter = PainterSpadeDecoder(opts)
            print("  - Created PainterSpadeDecoder Painter")
        else:
            self.painter = nn.Module()
            print("  - Created Empty Painter")

    def encode(self, x):
        assert self.encoder is not None
        return self.encoder.forward(x)

    def __str__(self):
        return strings.generator(self)


class MaskDecoder(BaseDecoder):
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


class DepthDecoder(nn.Module):
    """ Depth decoder based on depth auxiliary task in DADA paper
        Source: https://github.com/valeoai/DADA
    """

    def __init__(self, opts):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.enc4_1 = nn.Conv2d(
            2048, 512, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_size = opts.data.transforms[-1].new_size
        self.use_dada = False
        if opts.gen.d.use_dada:
            self.use_dada = True
            self.dec4 = nn.Conv2d(
                128, 2048, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        x = self.enc4_1(x)
        x = self.relu(x)
        x = self.enc4_2(x)
        x = self.relu(x)
        x_enc = self.enc4_3(x)

        depth = torch.mean(x_enc, dim=1, keepdim=True)
        depth = F.interpolate(
            depth,
            size=(384, 384),  # size used in MiDaS inference
            mode="bicubic",  # what MiDaS uses
            align_corners=False,
        )
        depth = F.interpolate(
            depth, (self.output_size, self.output_size), mode="nearest"
        )  # what we used in the transforms to resize input

        if self.use_dada:
            z_depth = self.dec4(x_enc)
            z_depth = self.relu(z_depth)
            return depth, z_depth

        return depth

    def __str__(self):
        return strings.basedecoder(self)


class SegmentationDecoder(BaseDecoder):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py
    def __init__(self, opts):
        super().__init__()
        self.aspp = ASPP(16, nn.BatchNorm2d)
        conv_modules = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        ]
        if opts.gen.s.upsample_featuremaps:
            conv_modules = [InterpolateNearest2d(scale_factor=2)] + conv_modules

        conv_modules += [
            nn.Conv2d(256, opts.gen.s.output_dim, kernel_size=1, stride=1),
        ]
        self.conv = nn.Sequential(*conv_modules)
        self._target_size = None

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
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate logits to the target seg map's size"
            raise Exception(error)
        if z.shape[1] != 2048:
            raise Exception(
                "Segmentation decoder will only work with 2048 channels for z"
            )
        y = self.aspp(z)
        y = self.conv(y)
        return F.interpolate(y, self._target_size, mode="bilinear", align_corners=True)
