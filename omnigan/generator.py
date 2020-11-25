"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
from omnigan.deeplabv3 import build_backbone, DeepLabV3Decoder
from omnigan.deeplabv2 import DeepLabV2Decoder
import torch.nn as nn
import torch
from omnigan.tutils import init_weights
from omnigan.blocks import (
    PainterSpadeDecoder,
    BaseDecoder,
    DADADepthRegressionDecoder,
)
from omnigan.encoder import DeeplabV2Encoder, BaseEncoder
import omnigan.strings as strings


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
    def __init__(self, opts, latent_shape=None, verbose=0, no_init=False):
        """Creates the generator. All decoders listed in opts.gen will be added
        to the Generator.decoders ModuleDict if opts.gen.DecoderInitial is not True.
        Then can be accessed as G.decoders.T or G.decoders["T"] for instance,
        for the image Translation decoder

        Args:
            opts (addict.Dict): configuration dict
        """
        super().__init__()
        self.opts = opts
        self.verbose = verbose

        self.encoder = None
        if any(t in opts.tasks for t in "msd"):
            if opts.gen.encoder.architecture == "deeplabv2":
                self.encoder = DeeplabV2Encoder(opts, no_init, verbose)
                if self.verbose > 0:
                    print("  - Created Deeplabv2 Encoder")
            elif opts.gen.encoder.architecture == "deeplabv3":
                self.encoder = build_backbone(opts, no_init)
                if self.verbose > 0:
                    print(
                        "  - Created Deeplabv3 ({}) Encoder".format(
                            opts.gen.deeplabv3.backbone
                        )
                    )
            else:
                self.encoder = BaseEncoder(opts)
                if self.verbose > 0:
                    print("  - Created Base Encoder")

        self.decoders = {}

        if "d" in opts.tasks:
            if opts.gen.d.architecture == "base":
                self.decoders["d"] = BaseDepthDecoder(opts)
            else:
                self.decoders["d"] = DADADepthRegressionDecoder(opts)

            if self.verbose > 0:
                print(f"  - Created {self.decoders['d'].__class__.__name__}")

        if "s" in opts.tasks:
            if opts.gen.s.architecture == "deeplabv2":
                self.decoders["s"] = DeepLabV2Decoder(opts)
                if self.verbose > 0:
                    print("  - Created DeepLabV2Decoder")
            elif opts.gen.s.architecture == "deeplabv3":
                self.decoders["s"] = DeepLabV3Decoder(opts)
                if self.verbose > 0:
                    print("  - Created DeepLabV3Decoder")
            else:
                raise NotImplementedError(
                    "Unknown architecture {}".format(opts.gen.s.architecture)
                )

        if "m" in opts.tasks:
            if self.verbose > 0:
                print("  - Created Mask Decoder")
            self.decoders["m"] = MaskDecoder(opts)

        self.decoders = nn.ModuleDict(self.decoders)

        if "p" in self.opts.tasks:
            self.painter = PainterSpadeDecoder(opts)
            if self.verbose > 0:
                print("  - Created PainterSpadeDecoder Painter")
        else:
            self.painter = nn.Module()
            if self.verbose > 0:
                print("  - Created Empty Painter")

    def encode(self, x):
        assert self.encoder is not None
        return self.encoder.forward(x)

    def __str__(self):
        return strings.generator(self)

    def sample_painter_z(self, batch_size, device, force_half=False):
        if self.opts.gen.p.no_z:
            return None

        z = torch.empty(
            batch_size,
            self.opts.gen.p.latent_dim,
            self.painter.z_h,
            self.painter.z_w,
            device=device,
        ).normal_(mean=0, std=1.0)

        if force_half:
            z = z.half()

        return z

    def mask(self, x=None, z=None, sigmoid=True):
        assert x is not None or z is not None
        assert not (x is not None and z is not None)
        if z is None:
            z = self.encode(x)
        logits = self.decoders["m"](z)

        if not sigmoid:
            return logits

        return torch.sigmoid(logits)

    def paint(self, m, x):
        """
        Paints given a mask and an image
        calls painter(z, x * (1.0 - m))
        Mask has 1s where water should be painted

        Args:
            m (torch.Tensor): Mask
            x (torch.Tensor): Image to paint

        Returns:
            torch.Tensor: painted image
        """
        z_paint = self.sample_painter_z(x.shape[0], x.device)
        fake = self.painter(z_paint, x * (1.0 - m))
        if self.opts.gen.p.paste_original_content:
            return x * (1.0 - m) + fake * m
        return fake

    def depth_image(self, x=None, z=None):
        assert x is not None or z is not None
        assert not (x is not None and z is not None)
        if z is None:
            z = self.encode(x)
        logits = self.decoders["d"](z)

        if logits.shape[1] > 1:
            logits = torch.argmax(logits, dim=1)
            logits = logits / logits.max()

        return logits


class MaskDecoder(BaseDecoder):
    def __init__(self, opts):
        low_level_feats_dim = -1
        use_v3 = opts.gen.encoder.architecture == "deeplabv3"
        use_mobile_net = opts.gen.deeplabv3.backbone == "mobilenet"
        use_low = opts.gen.m.use_low_level_feats

        if use_v3 and use_mobile_net:
            input_dim = 320
            if use_low:
                low_level_feats_dim = 24
        elif use_v3:
            input_dim = 2048
            if use_low:
                low_level_feats_dim = 256
        else:
            input_dim = 2048

        super().__init__(
            n_upsample=opts.gen.m.n_upsample,
            n_res=opts.gen.m.n_res,
            input_dim=input_dim,
            proj_dim=opts.gen.m.proj_dim,
            output_dim=opts.gen.m.output_dim,
            res_norm=opts.gen.m.res_norm,
            activ=opts.gen.m.activ,
            pad_type=opts.gen.m.pad_type,
            output_activ="none",
            low_level_feats_dim=low_level_feats_dim,
        )


class BaseDepthDecoder(BaseDecoder):
    def __init__(self, opts):
        low_level_feats_dim = -1
        use_v3 = opts.gen.encoder.architecture == "deeplabv3"
        use_mobile_net = opts.gen.deeplabv3.backbone == "mobilenet"
        use_low = opts.gen.d.use_low_level_feats

        if use_v3 and use_mobile_net:
            input_dim = 320
            if use_low:
                low_level_feats_dim = 24
        elif use_v3:
            input_dim = 2048
            if use_low:
                low_level_feats_dim = 256
        else:
            input_dim = 2048

        n_upsample = 1 if opts.gen.d.upsample_featuremaps else 0
        output_dim = (
            1
            if not opts.gen.d.classify.enable
            else opts.gen.d.classify.linspace.buckets
        )

        super().__init__(
            n_upsample=n_upsample,
            n_res=opts.gen.d.n_res,
            input_dim=input_dim,
            proj_dim=opts.gen.d.proj_dim,
            output_dim=output_dim,
            res_norm=opts.gen.d.res_norm,
            activ=opts.gen.d.activ,
            pad_type=opts.gen.d.pad_type,
            output_activ="none",
            low_level_feats_dim=low_level_feats_dim,
        )
