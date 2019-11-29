import torch
import torch.nn as nn
import functools
from omnigan.utils import init_weights

# from torch.optim import lr_scheduler

# mainly from https://github.com/sangwoomo/instagan/blob/master/models/networks.py


def get_dis(opts):
    return OmniDiscriminator(opts)


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
    return init_net(net, init_type, init_gain, gpu_ids)


def get_norm_layer(norm_type="instance"):
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
    init_weights(net, init_type, init_gain=init_gain)
    return net


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + "_u")
            _ = getattr(self.module, self.name + "_v")
            _ = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


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


class OmniDiscriminator(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.A = self.T = None
        models = {}
        if "A" in opts.tasks:
            models["A"] = nn.ModuleDict(
                {
                    "r": define_D(
                        3,
                        opts.dis.A.ndf,
                        n_layers_D=opts.dis.A.n_layers,
                        norm=opts.dis.A.norm,
                        use_sigmoid=opts.dis.A.use_sigmoid,
                        init_type=opts.dis.A.init_type,
                        init_gain=opts.dis.A.init_gain,
                    ),
                    "f": define_D(
                        3,
                        opts.dis.A.ndf,
                        n_layers_D=opts.dis.A.n_layers,
                        norm=opts.dis.A.norm,
                        use_sigmoid=opts.dis.A.use_sigmoid,
                        init_type=opts.dis.A.init_type,
                        init_gain=opts.dis.A.init_gain,
                    ),
                }
            )
        if "T" in opts.tasks:
            models["T"] = nn.ModuleDict(
                {
                    "f": define_D(
                        3,
                        opts.dis.T.ndf,
                        n_layers_D=opts.dis.T.n_layers,
                        norm=opts.dis.T.norm,
                        use_sigmoid=opts.dis.T.use_sigmoid,
                        init_type=opts.dis.T.init_type,
                        init_gain=opts.dis.T.init_gain,
                    ),
                    "n": define_D(
                        3,
                        opts.dis.T.ndf,
                        n_layers_D=opts.dis.T.n_layers,
                        norm=opts.dis.T.norm,
                        use_sigmoid=opts.dis.T.use_sigmoid,
                        init_type=opts.dis.T.init_type,
                        init_gain=opts.dis.T.init_gain,
                    ),
                }
            )
        self.models = nn.ModuleDict(models)
