"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
import torch
import torch.nn as nn
from omnigan.tutils import init_weights, get_4D_bit, get_conditioning_tensor
from omnigan.blocks import Conv2dBlock, ResBlocks, SpadeDecoder, BaseDecoder
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
        self.encoder = Encoder(opts)
        self.verbose = verbose
        self.decoders = {}

        if "t" in opts.tasks and not opts.gen.t.ignore:
            if opts.gen.t.use_bit_conditioning or opts.gen.t.use_spade:
                self.decoders["t"] = None
                # call set_translation_decoder(latent_shape, device)
            else:
                self.decoders["t"] = nn.ModuleDict(
                    {
                        "f": BaseTranslationDecoder(opts),
                        "n": BaseTranslationDecoder(opts),
                    }
                )

        if "a" in opts.tasks and not opts.gen.a.ignore:
            self.decoders["a"] = nn.ModuleDict(
                {"r": AdaptationDecoder(opts), "s": AdaptationDecoder(opts)}
            )

        if "d" in opts.tasks and not opts.gen.d.ignore:
            self.decoders["d"] = DepthDecoder(opts)

        if "h" in opts.tasks and not opts.gen.h.ignore:
            self.decoders["h"] = HeightDecoder(opts)

        if "s" in opts.tasks and not opts.gen.s.ignore:
            self.decoders["s"] = SegmentationDecoder(opts)

        if "w" in opts.tasks and not opts.gen.w.ignore:
            self.decoders["w"] = WaterDecoder(opts)

        self.decoders = nn.ModuleDict(self.decoders)

    def set_translation_decoder(self, latent_shape, device):
        if self.opts.gen.t.use_bit_conditioning:
            if not self.opts.gen.t.use_spade:
                raise ValueError("cannot have use_bit_conditioning but not use_spade")
            self.decoders["t"] = SpadeTranslationDict(latent_shape, self.opts)
            self.decoders["t"] = self.decoders["t"].to(device)
        elif self.opts.gen.t.use_spade:
            self.decoders["t"] = nn.ModuleDict(
                {
                    "f": SpadeTranslationDecoder(latent_shape, self.opts).to(device),
                    "n": SpadeTranslationDecoder(latent_shape, self.opts).to(device),
                }
            )
        for k in ["f", "n"]:
            init_weights(
                self.decoders["t"][k],
                init_type=self.opts.gen.t.init_type,
                init_gain=self.opts.gen.t.init_gain,
                verbose=self.verbose,
            )
        else:
            pass  # not using spade in anyway: do nothing

    def translate_batch(self, batch, translator="f", z=None):
        """Computes the translation of the images in a batch, according amongst
        other things to batch["domain"]

        Args:
            batch (dict): Batch dict with keys ['data', 'paths', 'domain', 'mode']
            translator (str, optional): Translation decoder to use. Defaults to "f".
            z (torch.Tensor, optional): Precomputed z. Defaults to None.

        Returns:
            torch.Tensor: 4D image tensor
        """
        x = batch["data"]["x"]
        if z is None:
            z = self.encode(x)

        K = None
        if self.opts.gen.t.use_spade:
            task_tensors = self.decode_tasks(z)
            K = get_conditioning_tensor(x, task_tensors)

        y = self.decoders["t"][translator](z, K)
        return y

    def decode_tasks(self, z):
        return {
            task: self.decoders[task](z)
            for task in self.opts.tasks
            if task not in {"t", "a"}
        }

    def encode(self, x):
        return self.encoder.forward(x)

    def forward_x(self, x, translator="f"):
        """Computes the translation of an image x to `translator`'s domain.
        Note this function will encode z and decode the necessary conditioning
        task tensors

        Args:
            x (torch.Tensor): images to translate
            translator (str, optional): translation translator to use. Defaults to "f".
            classifier_probs (list, optional): probabilities of belonging to a domain.
                Defaults to None.

        Returns:
            torch.Tensor: translated image
        """
        z = self.encode(x)
        cond = None
        if self.opts.gen.t.use_spade:
            task_tensors = self.decode_tasks(z)
            cond = get_conditioning_tensor(x, task_tensors)
        y = self.decoders["t"][translator](z, cond)
        return y

    def forward(self, x, translator="f"):
        return self.forward_x(x, translator)

    def __str__(self):
        return strings.generator(self)


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
        res_norm = opts.gen.encoder.res_norm
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
            ResBlocks(n_res, dim, norm=res_norm, activation=activ, pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return strings.encoder(self)


class HeightDecoder(BaseDecoder):
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


class WaterDecoder(BaseDecoder):
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


class DepthDecoder(BaseDecoder):
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


class SegmentationDecoder(BaseDecoder):
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


class AdaptationDecoder(BaseDecoder):
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

    def forward(self, z, cond=None):
        return self.model(z)


class BaseTranslationDecoder(BaseDecoder):
    def __init__(self, opts):
        super().__init__(
            opts.gen.t.n_upsample,
            opts.gen.t.n_res,
            opts.gen.t.res_dim,
            opts.gen.t.output_dim,
            res_norm=opts.gen.t.res_norm,
            activ=opts.gen.t.activ,
            pad_type=opts.gen.t.pad_type,
        )

    def forward(self, z, cond=None):
        return self.model(z)


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
