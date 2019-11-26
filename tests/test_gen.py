import numpy as np
import torch
from addict import Dict
import sys
sys.path.append("..")
import omnigan
from omnigan.generator import Generator

if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    conf = Dict()

    batch_size = 7
    latent_space_dims = [64, 32, 32]

    image = torch.randn(batch_size, 3, 256, 256)

    test_partial_decoder = True
    test_encoder = True
    test_encode_decode = True

    if test_partial_decoder:
        conf.gen.A.ignore = False
        conf.gen.D.ignore = True
        conf.gen.H.ignore = False
        conf.gen.T.ignore = False
        conf.gen.W.ignore = False
        G = Generator(conf)
        G.init_weights()
        print("D" in G.decoders)
        print("A" in G.decoders)
        x = torch.randn(batch_size, *latent_space_dims, dtype=torch.float32)
        v = G.decoders["W"](x)
        print(v.shape)
        print(sum(p.numel() for p in G.decoders.parameters()))

    conf.gen.A.ignore = False
    conf.gen.D.ignore = False
    conf.gen.H.ignore = False
    conf.gen.T.ignore = False
    conf.gen.W.ignore = False

    G = Generator(conf)
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
