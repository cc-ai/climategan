"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
import torch
import torch.nn as nn
from omnigan.tutils import init_weights, get_4D_bit, get_conditioning_tensor
from omnigan.blocks import Conv2dBlock, ResBlocks, SpadeDecoder, BaseDecoder
from omnigan.encoder import DeeplabEncoder, BaseEncoder
import omnigan.strings as strings

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------

# TODO think about how to use the classifier probs at inference


def get_gen(opts, latent_shape=None, verbose=0):
    G = OmniGenerator(opts, latent_shape, verbose)
    for model in G.decoders:
        if model == "t":
            if opts.gen.t.use_spade or opts.gen.t.use_bit_conditioning:
                continue
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

    def encode(self, x):
        return self.encoder.forward(x)

    def forward(self, x):
        return self.encode(x)

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


class SpadeTranslationDict(nn.ModuleDict):
    def __init__(self, latent_shape, opts):
        super().__init__()
        self.opts = opts
        self._model = SpadeTranslationDecoder(latent_shape, opts)

    def keys(self):
        return ["f", "n"]

    def __getitem__(self, key):
        self._model.update_bit(key)
        return self._model

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Cannot forward the SpadeTranslationDict, chose a domain"
        )

    def __str__(self):
        return str(self._model).strip()


class SpadeTranslationDecoder(SpadeDecoder):
    def __init__(self, latent_shape, opts):
        self.bit = None
        self.use_bit_conditioning = opts.gen.t.use_bit_conditioning

        cond_nc = 4  # 4 domains => 4-channel bitmap
        cond_nc = 2 if self.use_bit_conditioning else 0  # 2 domains => 2-channel bitmap
        if "d" in opts.tasks:
            cond_nc += 1
        if "h" in opts.tasks:
            cond_nc += 1
        if "s" in opts.tasks:
            cond_nc += opts.gen.s.num_classes
        if "w" in opts.tasks:
            cond_nc += 1
        self.cond_nc = cond_nc

        super().__init__(
            latent_shape,  # c x h x w of z
            cond_nc,  # number of channels in the conditioning tensor
            opts.gen.t.spade_n_up,  # number of upsampling
            opts.gen.t.spade_use_spectral_norm,  # use spectral norm in spade blocks?
            opts.gen.t.spade_param_free_norm,  # parameter-free norm in spade blocks
            opts.gen.t.spade_kernel_size,  # 3
        )
        self.register_buffer("f_bit", torch.tensor([1, 0]))
        self.register_buffer("n_bit", torch.tensor([0, 1]))

    def update_bit(self, key):
        if key == "f":
            self.bit = self.f_bit
        elif key == "n":
            self.bit = self.n_bit
        else:
            raise KeyError(f"update_bit: unknown key {key}")

    def concat_bit_to_seg(self, seg):
        bit = get_4D_bit(seg.shape, self.bit)
        return torch.cat(
            [bit.to(torch.float32).to(seg.device), seg.to(torch.float32)], dim=1
        )

    def forward(self, z, seg):
        if self.use_bit_conditioning:
            seg = self.concat_bit_to_seg(seg)
        return self._forward(z, seg)
