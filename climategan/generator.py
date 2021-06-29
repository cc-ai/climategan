"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
from pathlib import Path
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from torch import softmax

import climategan.strings as strings
from climategan.deeplab import create_encoder, create_segmentation_decoder
from climategan.depth import create_depth_decoder
from climategan.masker import create_mask_decoder
from climategan.painter import create_painter
from climategan.tutils import init_weights, mix_noise, normalize


def create_generator(opts, device="cpu", latent_shape=None, no_init=False, verbose=0):
    G = OmniGenerator(opts, latent_shape, verbose, no_init)
    if no_init:
        print("Sending to", device)
        return G.to(device)

    for model in G.decoders:
        net = G.decoders[model]
        if model == "s":
            continue
        if isinstance(net, nn.ModuleDict):
            for domain, domain_model in net.items():
                init_weights(
                    net[domain_model],
                    init_type=opts.gen[model].init_type,
                    init_gain=opts.gen[model].init_gain,
                    verbose=verbose,
                    caller=f"create_generator decoder {model} {domain}",
                )
        else:
            init_weights(
                G.decoders[model],
                init_type=opts.gen[model].init_type,
                init_gain=opts.gen[model].init_gain,
                verbose=verbose,
                caller=f"create_generator decoder {model}",
            )
    if G.encoder is not None and opts.gen.encoder.architecture == "base":
        init_weights(
            G.encoder,
            init_type=opts.gen.encoder.init_type,
            init_gain=opts.gen.encoder.init_gain,
            verbose=verbose,
            caller="create_generator encoder",
        )

    print("Sending to", device)
    return G.to(device)


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
            self.encoder = create_encoder(opts, no_init, verbose)

        self.decoders = {}
        self.painter = nn.Module()

        if "d" in opts.tasks:
            self.decoders["d"] = create_depth_decoder(opts, no_init, verbose)

            if self.verbose > 0:
                print(f"  - Add {self.decoders['d'].__class__.__name__}")

        if "s" in opts.tasks:
            self.decoders["s"] = create_segmentation_decoder(opts, no_init, verbose)

        if "m" in opts.tasks:
            self.decoders["m"] = create_mask_decoder(opts, no_init, verbose)

        self.decoders = nn.ModuleDict(self.decoders)

        if "p" in self.opts.tasks:
            self.painter = create_painter(opts, no_init, verbose)
        else:
            if self.verbose > 0:
                print("  - Add Empty Painter")

    def __str__(self):
        return strings.generator(self)

    def encode(self, x):
        """
        Forward x through the encoder

        Args:
            x (torch.Tensor): B3HW input tensor

        Returns:
            list: High and Low level features from the encoder
        """
        assert self.encoder is not None
        return self.encoder.forward(x)

    def decode(self, x=None, z=None, return_z=False, return_z_depth=False):
        """
        Comptutes the predictions of all available decoders from either x or z.
        If using spade for the masker with 15 channels, x *must* be provided,
        whether z is too or not.

        Args:
            x (torch.Tensor, optional): Input tensor (B3HW). Defaults to None.
            z (list, optional): List of high and low-level features as BCHW.
                Defaults to None.
            return_z (bool, optional): whether or not to return z in the dict.
                Defaults to False.
            return_z_depth (bool, optional): whether or not to return z_depth
                in the dict. Defaults to False.

        Raises:
            ValueError: If using spade for the masker with 15 channels but x is None

        Returns:
            dict: {task: prediction_tensor} (may include z and z_depth
                depending on args)
        """

        assert x is not None or z is not None
        if self.opts.gen.m.use_spade and self.opts.m.spade.cond_nc == 15:
            if x is None:
                raise ValueError(
                    "When using spade for the Masker with 15 channels,"
                    + " x MUST be provided"
                )

        z_depth = cond = d = s = None
        out = {}

        if z is None:
            z = self.encode(x)

        if return_z:
            out["z"] = z

        if "d" in self.decoders:
            d, z_depth = self.decoders["d"](z)
            out["d"] = d

        if return_z_depth:
            out["z_depth"] = z_depth

        if "s" in self.decoders:
            s = self.decoders["s"](z, z_depth)
            out["s"] = s

        if "m" in self.decoders:
            if s is not None and d is not None:
                cond = self.make_m_cond(d, s, x)
            m = self.mask(z=z, cond=cond)
            out["m"] = m

        return out

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

    def make_m_cond(self, d, s, x=None):
        """
        Create the masker's conditioning input when using spade from the
        d and s predictions and from the input x when cond_nc == 15.

        d and s are assumed to have the the same spatial resolution.
        if cond_nc == 15 then x is interpolated to match that dimension.

        Args:
            d (torch.Tensor): Raw depth prediction (B1HW)
            s (torch.Tensor): Raw segmentation prediction (BCHW)
            x (torch.Tensor, optional): Input tensor (B3hW). Mandatory
                when opts.gen.m.spade.cond_nc == 15

        Raises:
            ValueError: opts.gen.m.spade.cond_nc == 15 but x is None

        Returns:
            torch.Tensor: B x cond_nc x H  x W conditioning tensor.
        """
        if self.opts.gen.m.spade.detach:
            d = d.detach()
            s = s.detach()
        cats = [normalize(d), softmax(s, dim=1)]
        if self.opts.gen.m.spade.cond_nc == 15:
            if x is None:
                raise ValueError(
                    "When using spade for the Masker with 15 channels,"
                    + " x MUST be provided"
                )
            cats += [
                F.interpolate(x, s.shape[-2:], mode="bilinear", align_corners=True)
            ]

        return torch.cat(cats, dim=1)

    def mask(self, x=None, z=None, cond=None, z_depth=None, sigmoid=True):
        """
        Create a mask from either an input x or a latent vector z.
        Optionally if the Masker has a spade architecture the conditioning tensor
        may be provided (cond). Default behavior applies an element-wise
        sigmoid, but can be deactivated (sigmoid=False).

        At least one of x or z must be provided (i.e. not None).
        If the Masker has a spade architecture and cond_nc == 15 then x cannot
        be None.

        Args:
            x (torch.Tensor, optional): Input tensor B3HW. Defaults to None.
            z (list, optional): High and Low level features of the encoder.
                Will be computed if None. Defaults to None.
            cond ([type], optional): [description]. Defaults to None.
            sigmoid (bool, optional): [description]. Defaults to True.

        Returns:
            torch.Tensor: B1HW mask tensor
        """
        assert x is not None or z is not None
        if z is None:
            z = self.encode(x)

        if cond is None and self.opts.gen.m.use_spade:
            assert "s" in self.opts.tasks and "d" in self.opts.tasks
            with torch.no_grad():
                d_pred, z_d = self.decoders["d"](z)
                s_pred = self.decoders["s"](z, z_d)
                cond = self.make_m_cond(d_pred, s_pred, x)
        if z_depth is None and self.opts.gen.m.use_dada:
            assert "d" in self.opts.tasks
            with torch.no_grad():
                _, z_depth = self.decoders["d"](z)

        if cond is not None:
            device = z[0].device if isinstance(z, (tuple, list)) else z.device
            cond = cond.to(device)

        logits = self.decoders["m"](z, cond, z_depth)

        if not sigmoid:
            return logits

        return torch.sigmoid(logits)

    def paint(self, m, x, no_paste=False):
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
        m = m.to(x.dtype)
        fake = self.painter(z_paint, x * (1.0 - m))
        if self.opts.gen.p.paste_original_content and not no_paste:
            return x * (1.0 - m) + fake * m
        return fake

    def paint_cloudy(self, m, x, s, sky_idx=9, res=(8, 8), weight=0.8):
        """
        Paints x with water in m through an intermediary cloudy image
        where the sky has been replaced with perlin noise to imitate clouds.

        The intermediary cloudy image is only used to control the painter's
        painting mode, probing it with a cloudy input.

        Args:
            m (torch.Tensor): water mask
            x (torch.Tensor): input tensor
            s (torch.Tensor): segmentation prediction (BCHW)
            sky_idx (int, optional): Index of the sky class along s's C dimension.
                Defaults to 9.
            res (tuple, optional): Perlin noise spatial resolution. Defaults to (8, 8).
            weight (float, optional): Intermediate image's cloud proportion
                (w * cloud + (1-w) * original_sky). Defaults to 0.8.

        Returns:
            torch.Tensor: painted image with original content pasted.
        """
        sky_mask = (
            torch.argmax(
                F.interpolate(s, x.shape[-2:], mode="bilinear"), dim=1, keepdim=True
            )
            == sky_idx
        ).to(x.dtype)
        noised_x = mix_noise(x, sky_mask, res=res, weight=weight).to(x.dtype)
        fake = self.paint(m, noised_x, no_paste=True)
        return x * (1.0 - m) + fake * m

    def depth(self, x=None, z=None, return_z_depth=False):
        """
        Compute the depth head's output

        Args:
            x (torch.Tensor, optional): Input B3HW tensor. Defaults to None.
            z (list, optional): High and Low level features of the encoder.
                Defaults to None.

        Returns:
            torch.Tensor: B1HW tensor of depth predictions
        """
        assert x is not None or z is not None
        assert not (x is not None and z is not None)
        if z is None:
            z = self.encode(x)
        depth, z_depth = self.decoders["d"](z)

        if depth.shape[1] > 1:
            depth = torch.argmax(depth, dim=1)
            depth = depth / depth.max()

        if return_z_depth:
            return depth, z_depth

        return depth

    def load_val_painter(self):
        """
        Loads a validation painter if available in opts.val.val_painter

        Returns:
            bool: operation success status
        """
        try:
            # key exists in opts
            assert self.opts.val.val_painter

            # path exists
            ckpt_path = Path(self.opts.val.val_painter).resolve()
            assert ckpt_path.exists()

            # path is a checkpoint path
            assert ckpt_path.is_file()

            # opts are available in that path
            opts_path = ckpt_path.parent.parent / "opts.yaml"
            assert opts_path.exists()

            # load opts
            with opts_path.open("r") as f:
                val_painter_opts = Dict(yaml.safe_load(f))

            # load checkpoint
            state_dict = torch.load(ckpt_path)

            # create dummy painter from loaded opts
            painter = create_painter(val_painter_opts)

            # load state-dict in the dummy painter
            painter.load_state_dict(
                {k.replace("painter.", ""): v for k, v in state_dict["G"].items()}
            )

            # send to current device in evaluation mode
            device = next(self.parameters()).device
            self.painter = painter.eval().to(device)

            # disable gradients
            for p in self.painter.parameters():
                p.requires_grad = False

            # success
            print("    - Loaded validation-only painter")
            return True

        except Exception as e:
            # something happened, aborting gracefully
            print(traceback.format_exc())
            print(e)
            print(">>> WARNING: error (^) in load_val_painter, aborting.")
            return False
