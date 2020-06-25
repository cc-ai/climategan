"""Discriminator architecture for OmniGAN's GAN components (a and t)
"""
import torch
import torch.nn as nn
import functools
from omnigan.tutils import init_weights
from omnigan.blocks import SpectralNorm

# from torch.optim import lr_scheduler

# mainly from https://github.com/sangwoomo/instagan/blob/master/models/networks.py


def get_dis(opts, verbose):
    disc = OmniDiscriminator(opts)
    for task, model in disc.items():
        for domain_model in model.values():
            init_weights(
                domain_model,
                init_type=opts.dis[task].init_type,
                init_gain=opts.dis[task].init_gain,
                verbose=verbose,
            )
    return disc


def define_D(
    input_nc,
    ndf,
    n_layers_D=3,
    norm="batch",
    use_sigmoid=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(
        input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
    )
    return net


def get_norm_layer(norm_type="instance"):
    if not norm_type:
        print("norm_type is {}, defaulting to instance")
        norm_type = "instance"
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type=init_type, init_gain=init_gain)
    return net


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            # Use spectral normalization
            SpectralNorm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # Use spectral normalization
                SpectralNorm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # Use spectral normalization
            SpectralNorm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                )
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Use spectral normalization
        sequence += [
            SpectralNorm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class OmniDiscriminator(nn.ModuleDict):
    def __init__(self, opts):
        super().__init__()
        if "p" in opts.tasks:
            self["p"] = nn.ModuleDict(
                {
                    "global": define_D(
                        input_nc=3,
                        ndf=opts.dis.p.ndf,
                        n_layers_D=opts.dis.p.n_layers,
                        norm=opts.dis.p.norm,
                        use_sigmoid=opts.dis.p.use_sigmoid,
                        init_type=opts.dis.p.init_type,
                        init_gain=opts.dis.p.init_gain,
                    ),
                    "local": define_D(
                        input_nc=3,
                        ndf=opts.dis.p.ndf,
                        n_layers_D=opts.dis.p.n_layers,
                        norm=opts.dis.p.norm,
                        use_sigmoid=opts.dis.p.use_sigmoid,
                        init_type=opts.dis.p.init_type,
                        init_gain=opts.dis.p.init_gain,
                    ),
                }
            )
        if "m" in opts.tasks:
            self["m"] = nn.ModuleDict({"Advent": get_fc_discriminator()})


def get_fc_discriminator(num_classes=2, ndf=64):
    return torch.nn.Sequential(
        torch.nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        torch.nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        torch.nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        torch.nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        torch.nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )
