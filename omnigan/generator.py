"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
import torch.nn as nn
from omnigan.tutils import init_weights
from omnigan.blocks import SpadeDecoder, BaseDecoder, ASPP, DepthDecoder
from omnigan.encoder import DeeplabEncoder, BaseEncoder
import omnigan.strings as strings
import torch.nn.functional as F

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------

# TODO think about how to use the classifier probs at inference


def get_gen(opts, latent_shape=None, verbose=0, no_init=False):
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
    # Init painter weights
    if not no_init:
        init_weights(
            G.painter,
            init_type=opts.gen.p.init_type,
            init_gain=opts.gen.p.init_gain,
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
        if "m" in opts.tasks:
            if opts.gen.encoder.architecture == "deeplabv2":
                self.encoder = DeeplabEncoder(opts)
                print("  - Created Pretrained Deeplab Encoder")
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
            self.painter = FullSpadeGen(opts)
            print("  - Created FullSpade Painter")
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


# class DepthDecoder(BaseDecoder):
#     def __init__(self, opts):
#         super().__init__(
#             n_upsample=opts.gen.d.n_upsample,
#             n_res=opts.gen.d.n_res,
#             input_dim=opts.gen.encoder.res_dim,
#             proj_dim=opts.gen.d.proj_dim,
#             output_dim=opts.gen.d.output_dim,
#             res_norm=opts.gen.d.res_norm,
#             activ=opts.gen.d.activ,
#             pad_type=opts.gen.d.pad_type,
#             output_activ="sigmoid",
#             conv_norm=opts.gen.d.conv_norm,
#         )


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
            conv_modules = [nn.Upsample(scale_factor=2)] + conv_modules

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


class FullSpadeGen(nn.Module):
    def __init__(self, opts):
        super(FullSpadeGen, self).__init__()

        n_upsample = opts.gen.p.spade_n_up

        self.latent_dim = opts.gen.p.latent_dim
        self.batch_size = opts.data.loaders.batch_size

        # Get size of latent vector based on downsampling:
        self.dec = SpadeDecoder(
            latent_dim=self.latent_dim,
            cond_nc=3,
            spade_n_up=n_upsample,
            spade_use_spectral_norm=opts.gen.p.spade_use_spectral_norm,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

    def forward(self, z, cond):
        return self.dec(z, cond)
