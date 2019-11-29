import numpy as np
import torch
from addict import Dict
import sys

sys.path.append("..")
from omnigan.generator import OmniGenerator

if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    opts = Dict()
    opts.tasks = ["A", "D", "W", "H", "T"]

    batch_size = 7
    latent_space_dims = [64, 32, 32]

    image = torch.randn(batch_size, 3, 256, 256)

    test_partial_decoder = True
    test_encoder = True
    test_encode_decode = True

    if test_partial_decoder:
        opts.gen.A.ignore = False
        opts.gen.D.ignore = True
        opts.gen.H.ignore = False
        opts.gen.T.ignore = False
        opts.gen.W.ignore = False
        G = OmniGenerator(opts)
        G.init_weights()
        print("D" in G.decoders)
        print("A" in G.decoders)
        x = torch.randn(batch_size, *latent_space_dims, dtype=torch.float32)
        v = G.decoders["W"](x)
        print(v.shape)
        print(sum(p.numel() for p in G.decoders.parameters()))

    opts.gen.A.ignore = False
    opts.gen.D.ignore = False
    opts.gen.H.ignore = False
    opts.gen.T.ignore = False
    opts.gen.W.ignore = False

    G = OmniGenerator(opts)
    G.init_weights()
    print(G.decoders)
    print(G.E)

    if test_encoder:
        encoded = G.E(image)
        print(encoded.shape)

    if test_encode_decode:
        z = G.E(image)
        for dec in "ADTHW":
            if dec in G.decoders:
                if dec in "AT":
                    for d in G.decoders[dec]:
                        print(dec, d, G.decoders[dec][d](z).shape)
                else:
                    print(dec, G.decoders[dec](z).shape)
