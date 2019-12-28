import numpy as np
import torch
from addict import Dict
import sys

sys.path.append("..")

from omnigan.generator import get_gen, UnetEncoder, UnetDecoder
from omnigan.utils import load_opts
from run import print_header


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    opts = Dict()
    opts.tasks = ["a", "d", "w", "h", "t"]

    batch_size = 7
    latent_space_dims = [64, 32, 32]

    image = torch.randn(batch_size, 3, 256, 256)

    test_partial_decoder = True
    test_encoder = True
    test_encode_decode = True
    test_unets = True

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

    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")
    G = get_gen(opts)
    print("DECODERS:", G.decoders)
    print("ENCODER:", G.encoder)

    if test_encoder:
        print_header("test_encoder")
        encoded = G.encoder(image)
        print(encoded.shape)

    if test_encode_decode:
        print_header("test_encode_decode")
        z = G.encoder(image)
        for dec in "adhtw":
            if dec in G.decoders:
                if dec in "at":
                    for d in G.decoders[dec]:
                        print(dec, d, G.decoders[dec][d](z).shape)
                else:
                    print(dec, G.decoders[dec](z).shape)

    if test_unets:
        print_header("test_unets")
        x = torch.rand((2, 3, 128, 128))
        ue = UnetEncoder(opts)
        ud = UnetDecoder(opts, 3)
        z = ue(x)
        print(x.shape)
        print(len(z))
        print([tuple(a.shape) for a in z])
        print(ud(z).shape)
