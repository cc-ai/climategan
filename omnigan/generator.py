import torch.nn as nn
from omnigan.utils import init_weights
import numpy as np
import torch
import torch.nn.functional as F

# * https://docs.fast.ai/vision.models.unet.html
# * Dynamic UNet FastAI

# --------------------------------------------------------------------------
# -----  For now no network structure, just project in a 64 x 32 x 32  -----
# -----   latent space and decode to (3 or 1) x 256 x 256              -----
# --------------------------------------------------------------------------


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


class Encoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.project = nn.Conv2d(3, 64, 1)
        self.downsample = nn.AdaptiveMaxPool2d(32)

    def forward(self, x):
        return self.project(self.downsample(x))


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

        self.encoder = Encoder(opts)

        self.decoders = {}

        if "a" in opts.tasks and not opts.gen.A.ignore:
            self.decoders["a"] = nn.ModuleDict(
                {"r": AdapatationDecoder(opts), "s": AdapatationDecoder(opts)}
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


class Decoder(nn.Module):
    """generic class for decoders
    """

    def __init__(self, opts):
        super().__init__()

    def forward(self, x):
        return self.layers(x)


class HeightDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class WaterDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class DepthDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 1, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class TranslationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class SegmentationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, opts.gen.s.num_classes, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


class AdapatationDecoder(Decoder):
    def __init__(self, opts):
        super().__init__(opts)
        self.layers = []
        self.layers.append(nn.Conv2d(64, 3, 1))
        self.layers.append(nn.UpsamplingNearest2d(256))
        self.layers = nn.Sequential(*self.layers)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit
        #   /0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit
        #   /8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UnetEncoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        base_channels = opts.gen.encoder.base_channels
        n_blocks = opts.gen.encoder.n_blocks

        self.inc = DoubleConv(3, base_channels)

        self.down_channels = [
            (base_channels * 2 ** i, base_channels * 2 ** (i + 1))
            for i in range(n_blocks - 1)
        ]
        self.down_channels.append(
            (base_channels * 2 ** (n_blocks - 1), base_channels * 2 ** (n_blocks - 1))
        )
        self.downs = nn.Sequential(
            *[Down(inpc, outc) for inpc, outc in self.down_channels]
        )

    def forward(self, x):
        z = [self.inc(x)]
        for d in self.downs:
            z.append(d(z[-1]))
        return z


class UnetDecoder(nn.Module):
    def __init__(self, opts, output_channels):
        super().__init__()
        base_channels = opts.gen.encoder.base_channels
        n_blocks = opts.gen.encoder.n_blocks
        bilinear = opts.gen.encoder.bilinear
        self.up_channels = [
            (
                base_channels * 2 ** (n_blocks - i),
                base_channels * 2 ** (n_blocks - i - 2),
            )
            for i in range(n_blocks - 1)
        ]
        self.up_channels.append((base_channels * 2, base_channels))
        self.ups = nn.Sequential(
            *[Up(inpc, outc, bilinear) for inpc, outc in self.up_channels]
        )
        self.outc = OutConv(64, output_channels)

    def forward(self, z):
        y = self.ups[0](z[-1], z[-2])
        for i, up in enumerate(self.ups[1:]):
            y = up(y, z[-(i + 3)])
        logits = self.outc(y)
        return logits
