"""Discriminator architecture for OmniGAN's GAN components (a and t)
"""
import torch
import torch.nn as nn
import functools
from omnigan.tutils import init_weights
from omnigan.blocks import SpectralNorm

# from torch.optim import lr_scheduler

# mainly from https://github.com/sangwoomo/instagan/blob/master/models/networks.py


def get_dis(opts, verbose, no_init=False):
    disc = OmniDiscriminator(opts)
    if no_init:
        return disc

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
    get_intermediate_features=False,
    num_D=1,
):
    norm_layer = get_norm_layer(norm_type=norm)
    net = MultiscaleDiscriminator(
        input_nc,
        ndf,
        n_layers=3,
        norm_layer=norm_layer,
        use_sigmoid=use_sigmoid,
        get_intermediate_features=get_intermediate_features,
        num_D=num_D,
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
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        get_intermediate_features=True,
    ):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.get_intermediate_features = get_intermediate_features

        kw = 4
        padw = 1
        sequence = [
            [
                # Use spectral normalization
                SpectralNorm(
                    nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
                ),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                [
                    # Use spectral normalization
                    SpectralNorm(  # TODO replace with Conv2dBlock
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
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            [
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
        ]

        # Use spectral normalization
        sequence += [
            [
                SpectralNorm(
                    nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
                )
            ]
        ]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module("model" + str(n), nn.Sequential(*sequence[n]))
        # self.model = nn.Sequential(*sequence)

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = self.get_intermediate_features
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


#    def forward(self, input):
#        return self.model(input)


# Source: https://github.com/NVIDIA/pix2pixHD
class MultiscaleDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        get_intermediate_features=True,
        num_D=3,
    ):
        super(MultiscaleDiscriminator, self).__init__()
        # self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
        #         use_sigmoid=False, num_D=3, getIntermFeat=False

        self.n_layers = n_layers
        self.ndf = ndf
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        self.get_intermediate_features = get_intermediate_features
        self.num_D = num_D

        for i in range(self.num_D):
            netD = NLayerDiscriminator(
                input_nc=input_nc,
                ndf=self.ndf,
                n_layers=self.n_layers,
                norm_layer=self.norm_layer,
                use_sigmoid=self.use_sigmoid,
                get_intermediate_features=self.get_intermediate_features,
            )
            self.add_module("discriminator_%d" % i, netD)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, input):
        result = []
        get_intermediate_features = self.get_intermediate_features
        for name, D in self.named_children():
            if "discriminator" not in name:
                continue
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


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
                        get_intermediate_features=opts.dis.p.get_intermediate_features,
                        num_D=opts.dis.p.num_D,
                    ),
                    "local": define_D(
                        input_nc=3,
                        ndf=opts.dis.p.ndf,
                        n_layers_D=opts.dis.p.n_layers,
                        norm=opts.dis.p.norm,
                        use_sigmoid=opts.dis.p.use_sigmoid,
                        get_intermediate_features=opts.dis.p.get_intermediate_features,
                        num_D=opts.dis.p.num_D,
                    ),
                }
            )
        if "m" in opts.tasks:
            if opts.gen.m.use_advent:
                if opts.dis.m.architecture == "base":
                    self["m"] = nn.ModuleDict(
                        {"Advent": get_fc_discriminator(num_classes=2)}
                    )
                elif opts.dis.m.architecture == "OmniDiscriminator":
                    self["m"] = nn.ModuleDict(
                        {
                            "Advent": define_D(
                                input_nc=2,
                                ndf=opts.dis.m.ndf,
                                n_layers_D=opts.dis.m.n_layers,
                                norm=opts.dis.m.norm,
                                use_sigmoid=opts.dis.m.use_sigmoid,
                                get_intermediate_features=opts.dis.m.get_intermediate_features,
                                num_D=opts.dis.m.num_D,
                            )
                        }
                    )
                else:
                    raise Exception("This Discriminator is currently not supported!")
        if "s" in opts.tasks:
            if opts.gen.s.use_advent:
                self["s"] = nn.ModuleDict(
                    {"Advent": get_fc_discriminator(num_classes=11)}
                )


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
