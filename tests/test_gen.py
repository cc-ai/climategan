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

    test_partial_decoder = False
    print_architecture = False
    test_encoder = False
    test_encode_decode = False
    test_translation = False
    test_summary = True

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
        v = G.decoders["s"](x)
        print(v.shape)
        print(sum(p.numel() for p in G.decoders.parameters()))

    opts.gen.a.ignore = False
    opts.gen.d.ignore = False
    opts.gen.h.ignore = False
    opts.gen.t.ignore = False
    opts.gen.w.ignore = False

    G = get_gen(opts)
    if print_architecture:
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
        print(G.forward(image, translator="f").shape)

    if test_summary:
        print_header("Generator summary")
        print(summary(G, input_size=(3, 256, 256)))
