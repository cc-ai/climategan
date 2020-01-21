import torch
import torch.nn as nn
from omnigan.utils import init_weights, get_4D_bit, domains_to_class_tensor
from omnigan.blocks import Conv2dBlock, ResBlocks, SpadeDecoder, BaseDecoder

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------

# TODO think about how to use the classifier probs at inference


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
        self.opts = opts
        self.encoder = Encoder(opts)

        self.decoders = {}

        TranslationDecoder = (
            SpadeTranslationDecoder
            if self.opts.gen.t.use_spade
            else BaseTranslationDecoder
        )

        if "a" in opts.tasks and not opts.gen.A.ignore:
            self.decoders["a"] = nn.ModuleDict(
                {"r": AdaptationDecoder(opts), "s": AdaptationDecoder(opts)}
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

    def get_conditioning_tensor(self, x, task_tensors, classifier_probs=None):
        """creates the 4D tensor to condition the translation on by concatenating d, h, s, w
        and a conditioning bit:

        Args:
            task_tensors (torch.Tensor): dictionnary task: conditioning tensor
            classifier_probs (list, optional): 1-hot encoded depending on the
                domain to use. Defaults to None.

        Returns:
            torch.Tensor: conditioning tensor, all tensors concatenated
                on the channel dim
        """
        if classifier_probs is None:
            classifier_probs = torch.Tensor([0, 1, 0, 0]).detach().to(torch.float32)
        bit = get_4D_bit(x.shape, classifier_probs).detach().to(x.device)
        # bit => batchsize * conditioning tensor
        # conditioning tensor => 4 x h x d, with 0s or 1s as classifier_probs
        return torch.cat(list(task_tensors.values()) + [bit], dim=1)

    def translate(self, batch, translator="f", z=None):
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
            z = self.encoder(x)

        cond = None
        if self.opts.gen.t.use_spade:
            task_tensors = self.decode_tasks(z)

            classifier_probs = domains_to_class_tensor(batch["domain"], one_hot=True)
            cond = self.get_conditioning_tensor(x, task_tensors, classifier_probs)

        y = self.decoders["t"][translator](z, cond)
        return y

    def decode_tasks(self, z):
        return {
            task: self.decoders[task](z)
            for task in self.opts.tasks
            if task not in {"t", "a"}
        }

    def forward(self, x, translator="f", classifier_probs=[1, 0, 0, 0]):
        """Computes the translation of an image x to a flooding domain

        Args:
            x (torch.Tensor): images to translate
            translator (str, optional): translation translator to use. Defaults to "f".
            classifier_probs (list, optional): probabilities of belonging to a domain.
                Defaults to None.

        Returns:
            torch.Tensor: translated image
        """
        z = self.encoder(x)
        cond = None
        if self.opts.gen.t.use_spade:
            task_tensors = self.decode_tasks(z)
            classifier_probs = torch.tensor([classifier_probs for _ in range(len(x))])
            cond = self.get_conditioning_tensor(x, task_tensors, classifier_probs)
        y = self.decoders["t"][translator](z, cond)
        return y


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


class SpadeTranslationDecoder(SpadeDecoder):
    def __init__(self, opts):
        cond_nc = 4  # 4 domains => 4-channel bitmap
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
