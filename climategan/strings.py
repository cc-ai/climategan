"""custom __str__ methods for ClimateGAN's classes
"""
import torch
import torch.nn as nn


def title(name, color="\033[94m"):
    name = "====  " + name + "  ===="
    s = "=" * len(name)
    s = f"{s}\n{name}\n{s}"
    return f"\033[1m{color}{s}\033[0m"


def generator(G):
    s = title("OmniGenerator", "\033[95m") + "\n"

    s += str(G.encoder) + "\n\n"
    for d in G.decoders:
        if d not in {"a", "t"}:
            s += str(G.decoders[d]) + "\n\n"
        elif d == "a":
            s += "[r & s]\n" + str(G.decoders["a"]["r"]) + "\n\n"
        else:
            if G.opts.gen.t.use_bit_conditioning:
                s += "[bit]\n" + str(G.decoders["t"]) + "\n\n"
            else:
                s += "[f & n]\n" + str(G.decoders["t"]["f"]) + "\n\n"
    return s.strip()


def encoder(E):
    s = title("Encoder") + "\n"
    for b in E.model:
        s += str(b) + "\n"
    return s.strip()


def get_conv_weight(conv):
    weight = torch.Tensor(
        conv.out_channels, conv.in_channels // conv.groups, *conv.kernel_size
    )
    return weight.shape


def conv2dblock(obj):
    name = "{:20}".format("Conv2dBlock")
    s = ""
    if "SpectralNorm" in obj.conv.__class__.__name__:
        s = "SpectralNorm => "
        w = str(tuple(get_conv_weight(obj.conv.module)))
    else:
        w = str(tuple(get_conv_weight(obj.conv)))
    return f"{name}{s}{w}".strip()


def resblocks(rb):
    s = "{}\n".format(f"ResBlocks({len(rb.model)})")
    for i, r in enumerate(rb.model):
        s += f"  - ({i}) {str(r)}\n"
    return s.strip()


def resblock(rb):
    s = "{:12}".format("Resblock")
    return f"{s}{rb.dim} channels, {rb.norm} norm + {rb.activation}"


def basedecoder(bd):
    s = title(bd.__class__.__name__) + "\n"
    for b in bd.model:
        if isinstance(b, nn.Upsample) or "InterpolateNearest2d" in b.__class__.__name__:
            s += "{:20}".format("Upsample") + "x2\n"
        else:
            s += str(b) + "\n"
    return s.strip()


def spaderesblock(srb):
    name = "{:20}".format("SPADEResnetBlock") + f"k {srb.kernel_size}, "
    s = f"{name}{srb.fin} > {srb.fout}, "
    s += f"param_free_norm: {srb.param_free_norm}, "
    s += f"spectral_norm: {srb.use_spectral_norm}"
    return s.strip()


def spadedecoder(sd):
    s = title(sd.__class__.__name__) + "\n"
    up = "{:20}x2\n".format("Upsample")
    s += up
    s += str(sd.head_0) + "\n"
    s += up
    s += str(sd.G_middle_0) + "\n"
    s += up
    s += str(sd.G_middle_1) + "\n"
    for i, u in enumerate(sd.up_spades):
        s += up
        s += str(u) + "\n"
    s += "{:20}".format("Conv2d") + str(tuple(get_conv_weight(sd.conv_img))) + " tanh"
    return s
