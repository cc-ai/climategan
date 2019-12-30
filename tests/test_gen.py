import numpy as np
import torch
import sys
from torchsummary import summary

sys.path.append("..")

from omnigan.generator import get_gen
from omnigan.utils import load_opts
from run import print_header


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")
    batch_size = 2
    latent_space_dims = [256, 32, 32]

    image = torch.Tensor(batch_size, 3, 256, 256).uniform_(-1, 1)

    test_partial_decoder = True
    test_encoder = True
    test_encode_decode = True
    test_translation = True
    test_summary = False

    if test_partial_decoder:
        print_header("test_partial_decoder")
        opts.gen.a.ignore = False
        opts.gen.d.ignore = True
        opts.gen.h.ignore = False
        opts.gen.t.ignore = False
        opts.gen.w.ignore = False
        G = get_gen(opts)
        print("d" in G.decoders)
        print("a" in G.decoders)
        x = torch.randn(batch_size, *latent_space_dims, dtype=torch.float32)
        v = G.decoders["w"](x)
        print(v.shape)
        print(sum(p.numel() for p in G.decoders.parameters()))

    opts.gen.a.ignore = False
    opts.gen.d.ignore = False
    opts.gen.h.ignore = False
    opts.gen.t.ignore = False
    opts.gen.w.ignore = False

    G = get_gen(opts)
    print("DECODERS:", G.decoders)
    print("ENCODER:", G.encoder)

    if test_encoder:
        print_header("test_encoder")
        encoded = G.encoder(image)
        print("Latent space dims {}".format(tuple(encoded.shape)[1:]))

    if test_encode_decode:
        print_header("test_encode_decode")
        z = G.encoder(image)
        for dec in "adhtw":
            if dec in G.decoders:
                if dec == "t":
                    continue
                if dec == "a":
                    for d in G.decoders[dec]:
                        print(dec, d, G.decoders[dec][d](z).shape)
                else:
                    print(dec, G.decoders[dec](z).shape)

    if test_translation:
        print_header("test_translation")
        print("Encoding...", end="")
        z = G.encoder(image)
        print("Ok.")
        print("Decoding tasks...", end="")
        h = G.decoders["h"](z)
        d = G.decoders["d"](z)
        s = G.decoders["s"](z)
        w = G.decoders["w"](z)
        print("h, d, s, w: Ok.")
        print("Translating...", end="")
        im = G.decoders["t"]["f"](z, torch.cat([h, d, s, w], dim=1))
        print("Decoded.")

    if test_summary:
        print_header("Generator summary")
        print(summary(G, input_size=(3, 256, 256)))
