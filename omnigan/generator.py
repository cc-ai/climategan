"""Complete Generator architecture:
    * OmniGenerator
    * Encoder
    * Decoders
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from torch import softmax

import omnigan.strings as strings
from omnigan.deeplab import create_encoder, create_segmentation_decoder
from omnigan.depth import create_depth_decoder
from omnigan.masker import create_mask_decoder
from omnigan.painter import create_painter
from omnigan.tutils import init_weights, mix_noise, normalize


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
            caller=f"create_generator encoder",
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

    def make_m_cond(self, d, s, x):
        if self.opts.gen.m.spade.detach:
            d = d.detach()
            s = s.detach()
        cats = [normalize(d), softmax(s, dim=1)]
        if self.opts.gen.m.spade.cond_nc == 15:
            assert x is not None
            cats += [
                F.interpolate(x, s.shape[-2:], mode="bilinear", align_corners=True)
            ]

        return torch.cat(cats, dim=1)

    def mask(self, x=None, z=None, cond=None, sigmoid=True):
        assert x is not None or z is not None
        assert not (x is not None and z is not None)
        if z is None:
            z = self.encode(x)

        if cond is None and self.opts.gen.m.use_spade:
            assert "s" in self.opts.tasks and "d" in self.opts.tasks
            with torch.no_grad():
                d_pred, z_depth = self.decoders["d"](z)
                s_pred = self.decoders["s"](z, z_depth)
                cond = self.make_m_cond(d_pred, s_pred, x)

        if cond is not None:
            device = z[0].device if isinstance(z, (tuple, list)) else z.device
            cond = cond.to(device)

        logits = self.decoders["m"](z, cond)

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
        sky_mask = (
            torch.argmax(
                F.interpolate(s, x.shape[-2:], mode="bilinear"), dim=1, keepdim=True
            )
            == sky_idx
        ).to(x.dtype)
        noised_x = mix_noise(x, sky_mask, res=res, weight=weight).to(x.dtype)
        fake = self.paint(m, noised_x, no_paste=True)
        return x * (1.0 - m) + fake * m

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

    def load_val_painter(self):
        try:
            assert self.opts.val.val_painter
            ckpt_path = Path(self.opts.val.val_painter).resolve()
            assert ckpt_path.exists()
            opts_path = ckpt_path.parent.parent / "opts.yaml"
            assert opts_path.exists()
            with opts_path.open("r") as f:
                val_painter_opts = Dict(yaml.safe_load(f))
            state_dict = torch.load(ckpt_path)
            painter = create_painter(val_painter_opts)
            painter.load_state_dict(
                {k.replace("painter.", ""): v for k, v in state_dict["G"].items()}
            )
            device = next(self.parameters()).device
            self.painter = painter.to(device)
            print("    - Loaded validation-only painter")
            return True
        except Exception as e:
            print(e)
            print(">>> WARNINT: error (^) in load_val_painter, aborting.")
            return False
