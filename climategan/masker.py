import torch
import torch.nn as nn
import torch.nn.functional as F

from climategan.blocks import (
    BaseDecoder,
    Conv2dBlock,
    InterpolateNearest2d,
    SPADEResnetBlock,
)


def create_mask_decoder(opts, no_init=False, verbose=0):
    if opts.gen.m.use_spade:
        if verbose > 0:
            print("  - Add Spade Mask Decoder")
        assert "d" in opts.tasks or "s" in opts.tasks
        return MaskSpadeDecoder(opts)
    else:
        if verbose > 0:
            print("  - Add Base Mask Decoder")
        return MaskBaseDecoder(opts)


class MaskBaseDecoder(BaseDecoder):
    def __init__(self, opts):
        low_level_feats_dim = -1
        use_v3 = opts.gen.encoder.architecture == "deeplabv3"
        use_mobile_net = opts.gen.deeplabv3.backbone == "mobilenet"
        use_low = opts.gen.m.use_low_level_feats
        use_dada = ("d" in opts.tasks) and opts.gen.m.use_dada

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
            norm=opts.gen.m.norm,
            activ=opts.gen.m.activ,
            pad_type=opts.gen.m.pad_type,
            output_activ="none",
            low_level_feats_dim=low_level_feats_dim,
            use_dada=use_dada,
        )


class MaskSpadeDecoder(nn.Module):
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
        self.opts = opts
        latent_dim = opts.gen.m.spade.latent_dim
        cond_nc = opts.gen.m.spade.cond_nc
        spade_use_spectral_norm = opts.gen.m.spade.spade_use_spectral_norm
        spade_param_free_norm = opts.gen.m.spade.spade_param_free_norm
        if self.opts.gen.m.spade.activations.all_lrelu:
            spade_activation = "lrelu"
        else:
            spade_activation = None
        spade_kernel_size = 3
        self.num_layers = opts.gen.m.spade.num_layers
        self.z_nc = latent_dim

        if (
            opts.gen.encoder.architecture == "deeplabv3"
            and opts.gen.deeplabv3.backbone == "mobilenet"
        ):
            self.input_dim = [320, 24]
            self.low_level_conv = Conv2dBlock(
                self.input_dim[1],
                self.input_dim[0],
                3,
                padding=1,
                activation="lrelu",
                pad_type="reflect",
                norm="spectral_batch",
            )
            self.merge_feats_conv = Conv2dBlock(
                self.input_dim[0] * 2,
                self.z_nc,
                3,
                padding=1,
                activation="lrelu",
                pad_type="reflect",
                norm="spectral_batch",
            )
        elif (
            opts.gen.encoder.architecture == "deeplabv3"
            and opts.gen.deeplabv3.backbone == "resnet"
        ):
            self.input_dim = [2048, 256]
            if self.opts.gen.m.use_proj:
                proj_dim = self.opts.gen.m.proj_dim
                self.low_level_conv = Conv2dBlock(
                    self.input_dim[1],
                    proj_dim,
                    3,
                    padding=1,
                    activation="lrelu",
                    pad_type="reflect",
                    norm="spectral_batch",
                )
                self.high_level_conv = Conv2dBlock(
                    self.input_dim[0],
                    proj_dim,
                    3,
                    padding=1,
                    activation="lrelu",
                    pad_type="reflect",
                    norm="spectral_batch",
                )
                self.merge_feats_conv = Conv2dBlock(
                    proj_dim * 2,
                    self.z_nc,
                    3,
                    padding=1,
                    activation="lrelu",
                    pad_type="reflect",
                    norm="spectral_batch",
                )
            else:
                self.low_level_conv = Conv2dBlock(
                    self.input_dim[1],
                    self.input_dim[0],
                    3,
                    padding=1,
                    activation="lrelu",
                    pad_type="reflect",
                    norm="spectral_batch",
                )
                self.merge_feats_conv = Conv2dBlock(
                    self.input_dim[0] * 2,
                    self.z_nc,
                    3,
                    padding=1,
                    activation="lrelu",
                    pad_type="reflect",
                    norm="spectral_batch",
                )

        elif opts.gen.encoder.architecture == "deeplabv2":
            self.input_dim = 2048
            self.fc_conv = Conv2dBlock(
                self.input_dim,
                self.z_nc,
                3,
                padding=1,
                activation="lrelu",
                pad_type="reflect",
                norm="spectral_batch",
            )
        else:
            raise ValueError("Unknown encoder type")

        self.spade_blocks = []

        for i in range(self.num_layers):
            self.spade_blocks.append(
                SPADEResnetBlock(
                    int(self.z_nc / (2 ** i)),
                    int(self.z_nc / (2 ** (i + 1))),
                    cond_nc,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                    spade_activation,
                ).cuda()
            )
        self.spade_blocks = nn.Sequential(*self.spade_blocks)

        self.final_nc = int(self.z_nc / (2 ** self.num_layers))
        self.mask_conv = Conv2dBlock(
            self.final_nc,
            1,
            3,
            padding=1,
            activation="none",
            pad_type="reflect",
            norm="spectral",
        )
        self.upsample = InterpolateNearest2d(scale_factor=2)

    def forward(self, z, cond, z_depth=None):
        if isinstance(z, (list, tuple)):
            z_h, z_l = z
            if self.opts.gen.m.use_proj:
                z_l = self.low_level_conv(z_l)
                z_l = F.interpolate(z_l, size=z_h.shape[-2:], mode="bilinear")
                z_h = self.high_level_conv(z_h)
            else:
                z_l = self.low_level_conv(z_l)
                z_l = F.interpolate(z_l, size=z_h.shape[-2:], mode="bilinear")
            z = torch.cat([z_h, z_l], axis=1)
            y = self.merge_feats_conv(z)
        else:
            y = self.fc_conv(z)

        for i in range(self.num_layers):
            y = self.spade_blocks[i](y, cond)
            y = self.upsample(y)
        y = self.mask_conv(y)
        return y

    def __str__(self):
        return "MaskerSpadeDecoder"
