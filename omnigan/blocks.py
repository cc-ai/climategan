import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

# TODO: Organise file


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm="in", activation="relu", pad_type="zero"):
        super().__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super().__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "instance":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "layer":
            self.norm = LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "spectral":
            self.norm = spectral_norm
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == "spectral":
            self.conv = SpectralNorm(
                nn.Conv2d(
                    input_dim, output_dim, kernel_size, stride, bias=self.use_bias
                )
            )
        else:
            self.conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride, bias=self.use_bias
            )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4
            # than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks"
    by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

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

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")  # noqa: F841
            v = getattr(self.module, self.name + "_v")  # noqa: F841
            w = getattr(self.module, self.name + "_bar")  # noqa: F841
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


class SpadeResBlocks(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        spade_use_spectral_norm,
        spade_param_free_norm,
        spade_kernel_size,
        cond_nc,
    ):
        super(SpadeResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                SPADEResnetBlock(
                    dim,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                    cond_nc,
                )
            ]
        self.model = nn.Sequential(*self.model)
        self.num_blocks = num_blocks

    def forward(self, x, seg):
        for j in range(self.num_blocks):
            x = self.model[j](x, seg)
        return x


##################################################################################
# SPADE Normalization
##################################################################################
class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, kernel_size, norm_nc, cond_nc):
        super().__init__()

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == "syncbatch":
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, nhidden, kernel_size=kernel_size, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(
            nhidden, norm_nc, kernel_size=kernel_size, padding=pw
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


##################################################################################
# Spade Blocks
##################################################################################
class SPADEResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        spade_use_spectral_norm,
        spade_param_free_norm,
        spade_kernel_size,
        cond_nc,
    ):
        super().__init__()
        # Attributes
        fin = dim
        fout = dim

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spade_use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)
        self.norm_1 = SPADE(spade_param_free_norm, spade_kernel_size, fmiddle, cond_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.activation(self.norm_0(x, seg)))
        dx = self.conv_1(self.activation(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def activation(self, x):
        return F.leaky_relu(x, 2e-1)


class SpadeDecoder(nn.Module):
    def __init__(
        self,
        n_upsample,  # number of upsampling
        n_res,  # number of resblocks before upsampling
        res_dim,  # resblock dimension
        output_dim,  # number of channels in the output
        activ,  # activation function
        pad_type,  # padding type
        spade_use_spectral_norm,  # whether or not to use spectral norm in spade blocks
        spade_param_free_norm,  # parameter-free norm in spade blocks
        spade_kernel_size,  # 3
        cond_nc,  # number of channels in the conditioning tensor
    ):

        super().__init__()
        self.n_res = n_res
        self.model = []

        # SPADE residual blocks
        self.model += [
            SpadeResBlocks(
                n_res,
                res_dim,
                spade_use_spectral_norm,
                spade_param_free_norm,
                spade_kernel_size,
                cond_nc,
            )
        ]
        # UPSAMPLING blocks
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(
                    res_dim,
                    res_dim // 2,
                    5,
                    1,
                    2,
                    norm="layer",
                    activation=activ,
                    pad_type=pad_type,
                ),
            ]
            res_dim //= 2
        # Use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                res_dim,
                output_dim,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type="reflect",
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, seg):
        for j in range(len(self.model)):
            if j == 0:
                x = self.model[j].forward(x, seg)
            else:
                x = self.model[j].forward(x)
        return x


class BaseDecoder(nn.Module):
    def __init__(
        self,
        n_upsample,
        n_res,
        dim,
        output_dim,
        res_norm="instance",
        activ="relu",
        pad_type="zero",
    ):
        super().__init__()

        self.model = [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(
                    dim,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm="layer",
                    activation=activ,
                    pad_type=pad_type,
                ),
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                dim,
                output_dim,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type=pad_type,
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
