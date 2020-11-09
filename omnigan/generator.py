"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
from omnigan.deeplabv3 import build_backbone, DeepLabV3Decoder
from omnigan.deeplabv2 import DeepLabV2Decoder
import torch.nn as nn
from omnigan.tutils import init_weights
from omnigan.blocks import (
    PainterSpadeDecoder,
    BaseDecoder,
    DepthDecoder,
)
from omnigan.encoder import DeeplabV2Encoder, BaseEncoder
import omnigan.strings as strings

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----  latent space and decode to (3 or 1) x 256 x 256               -----
# --------------------------------------------------------------------------

# TODO think about how to use the classifier probs at inference


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
                    caller=f"get_gen decoder {model} {domain}",
                )
        else:
            init_weights(
                G.decoders[model],
                init_type=opts.gen[model].init_type,
                init_gain=opts.gen[model].init_gain,
                verbose=verbose,
                caller=f"get_gen decoder {model}",
            )
    if G.encoder is not None and opts.gen.encoder.architecture != "deeplabv2":
        init_weights(
            G.encoder,
            init_type=opts.gen.encoder.init_type,
            init_gain=opts.gen.encoder.init_gain,
            verbose=verbose,
            caller=f"get_gen encoder",
        )
    # Init painter weights
    init_weights(
        G.painter,
        init_type=opts.gen.p.init_type,
        init_gain=opts.gen.p.init_gain,
        verbose=verbose,
        caller=f"get_gen painter",
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
        if any(t in opts.tasks for t in "msd"):
            if opts.gen.encoder.architecture == "deeplabv2":
                self.encoder = DeeplabV2Encoder(opts, no_init)
                print("  - Created Deeplab Encoder")
            if opts.gen.encoder.architecture == "deeplabv3":
                self.encoder = build_backbone(opts, no_init)
                print(
                    "  - Created Deeplabv3 ({}) Encoder".format(
                        opts.gen.deeplabv3.backbone
                    )
                )
            else:
                self.encoder = BaseEncoder(opts)
                print("  - Created Base Encoder")

        self.verbose = verbose
        self.decoders = {}

        if "d" in opts.tasks and not opts.gen.d.ignore:
            self.decoders["d"] = DepthDecoder(opts)
            print("  - Created Depth Decoder")

        if "s" in opts.tasks and not opts.gen.s.ignore:
            if opts.gen.s.architecture == "deeplabv2":
                self.decoders["s"] = DeepLabV2Decoder(opts)
                print("  - Created DeepLabV2Decoder")
            elif opts.gen.s.architecture == "deeplabv3":
                self.decoders["s"] = DeepLabV3Decoder(opts)
                print("  - Created DeepLabV3Decoder")
            else:
                raise NotImplementedError(
                    "Unknown architecture {}".format(opts.gen.s.architecture)
                )

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
            output_activ="none",
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

