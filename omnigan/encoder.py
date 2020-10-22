import torch
import torch.nn as nn
from omnigan.blocks import Conv2dBlock, ResBlocks
from omnigan.deeplabv2 import ResNetMulti, Bottleneck


class BaseEncoder(nn.Module):
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

        self.model = nn.Sequential()
        self.model.add_module(
            "conv2d_init",
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type
            ),
        )

        # downsampling blocks
        for i in range(n_downsample):
            self.model.add_module(
                "conv2d_" + str(i),
                Conv2dBlock(
                    dim,
                    2 * dim,
                    4,
                    2,
                    1,
                    norm=norm,
                    activation=activ,
                    pad_type=pad_type,
                ),
            )
            dim *= 2
        # residual blocks
        self.model.add_module(
            "resblock",
            ResBlocks(n_res, dim, norm=res_norm, activation=activ, pad_type=pad_type),
        )
        # self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class DeeplabEncoder(nn.Module):
    def __init__(self, opts):
        """Deeplab architecture encoder

        """
        super().__init__()

        self.model = ResNetMulti(
            Bottleneck, opts.gen.deeplabv2.nblocks, opts.gen.encoder.n_res
        )
        if opts.gen.deeplabv2.use_pretrained:
            saved_state_dict = torch.load(opts.gen.deeplabv2.pretrained_model)
            new_params = self.model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split(".")
                if not i_parts[1] in ["layer5", "resblock"]:
                    new_params[".".join(i_parts[1:])] = saved_state_dict[i]
            self.model.load_state_dict(new_params)

    def forward(self, x):
        return self.model(x)
