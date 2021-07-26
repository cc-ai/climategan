import torch
import torch.nn as nn
import torch.nn.functional as F

import climategan.strings as strings
from climategan.blocks import InterpolateNearest2d, SPADEResnetBlock
from climategan.norms import SpectralNorm


def create_painter(opts, no_init=False, verbose=0):
    if verbose > 0:
        print("  - Add PainterSpadeDecoder Painter")
    return PainterSpadeDecoder(opts)


class PainterSpadeDecoder(nn.Module):
    def __init__(self, opts):
        """Create a SPADE-based decoder, which forwards z and the conditioning
        tensors seg (in the original paper, conditioning is on a semantic map only).
        All along, z is conditioned on seg. First 3 SpadeResblocks (SRB) do not shrink
        the channel dimension, and an upsampling is applied after each. Therefore
        2 upsamplings at this point. Then, for each remaining upsamplings
        (w.r.t. spade_n_up), the SRB shrinks channels by 2. Before final conv to get 3
        channels, the number of channels is therefore:
            final_nc = channels(z) * 2 ** (spade_n_up - 2)
        Args:
            latent_dim (tuple): z's shape (only the number of channels matters)
            cond_nc (int): conditioning tensor's expected number of channels
            spade_n_up (int): Number of total upsamplings from z
            spade_use_spectral_norm (bool): use spectral normalization?
            spade_param_free_norm (str): norm to use before SPADE de-normalization
            spade_kernel_size (int): SPADE conv layers' kernel size
        Returns:
            [type]: [description]
        """
        super().__init__()

        latent_dim = opts.gen.p.latent_dim
        cond_nc = 3
        spade_n_up = opts.gen.p.spade_n_up
        spade_use_spectral_norm = opts.gen.p.spade_use_spectral_norm
        spade_param_free_norm = opts.gen.p.spade_param_free_norm
        spade_kernel_size = 3

        self.z_nc = latent_dim
        self.spade_n_up = spade_n_up

        self.z_h = self.z_w = None

        self.fc = nn.Conv2d(3, latent_dim, 3, padding=1)
        self.head_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.G_middle_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.G_middle_1 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.up_spades = nn.Sequential(
            *[
                SPADEResnetBlock(
                    self.z_nc // 2 ** i,
                    self.z_nc // 2 ** (i + 1),
                    cond_nc,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                )
                for i in range(spade_n_up - 2)
            ]
        )

        self.final_nc = self.z_nc // 2 ** (spade_n_up - 2)

        self.final_spade = SPADEResnetBlock(
            self.final_nc,
            self.final_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.final_shortcut = None
        if opts.gen.p.use_final_shortcut:
            self.final_shortcut = nn.Sequential(
                *[
                    SpectralNorm(nn.Conv2d(self.final_nc, 3, 1)),
                    nn.BatchNorm2d(3),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        self.conv_img = nn.Conv2d(self.final_nc, 3, 3, padding=1)

        self.upsample = InterpolateNearest2d(scale_factor=2)

    def set_latent_shape(self, shape, is_input=True):
        """
        Sets the latent shape to start the upsampling from, i.e. z_h and z_w.
        If is_input is True, then this is the actual input shape which should
        be divided by 2 ** spade_n_up
        Otherwise, just sets z_h and z_w from shape[-2] and shape[-1]

        Args:
            shape (tuple): The shape to start sampling from.
            is_input (bool, optional): Whether to divide shape by 2 ** spade_n_up
        """
        if isinstance(shape, (list, tuple)):
            self.z_h = shape[-2]
            self.z_w = shape[-1]
        elif isinstance(shape, int):
            self.z_h = self.z_w = shape
        else:
            raise ValueError("Unknown shape type:", shape)

        if is_input:
            self.z_h = self.z_h // (2 ** self.spade_n_up)
            self.z_w = self.z_w // (2 ** self.spade_n_up)

    def _apply(self, fn):
        # print("Applying SpadeDecoder", fn)
        super()._apply(fn)
        # self.head_0 = fn(self.head_0)
        # self.G_middle_0 = fn(self.G_middle_0)
        # self.G_middle_1 = fn(self.G_middle_1)
        # for i, up in enumerate(self.up_spades):
        #     self.up_spades[i] = fn(up)
        # self.conv_img = fn(self.conv_img)
        return self

    def forward(self, z, cond):
        if z is None:
            assert self.z_h is not None and self.z_w is not None
            z = self.fc(F.interpolate(cond, size=(self.z_h, self.z_w)))
        y = self.head_0(z, cond)
        y = self.upsample(y)
        y = self.G_middle_0(y, cond)
        y = self.upsample(y)
        y = self.G_middle_1(y, cond)

        for i, up in enumerate(self.up_spades):
            y = self.upsample(y)
            y = up(y, cond)

        if self.final_shortcut is not None:
            cond = self.final_shortcut(y)
        y = self.final_spade(y, cond)
        y = self.conv_img(F.leaky_relu(y, 2e-1))
        y = torch.tanh(y)
        return y

    def __str__(self):
        return strings.spadedecoder(self)
