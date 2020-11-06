import argparse
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.generator import get_gen
from omnigan.generator import SpadeDecoder
from omnigan.utils import load_test_opts
from omnigan.tutils import get_num_params
from run import print_header


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/maskgen_v0.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    latent_dim = opts.gen.encoder.res_dim
    latent_space_dims = [latent_dim, 4, 4]
    image = torch.Tensor(batch_size, 3, 32, 32).uniform_(-1, 1).to(device)
    # -------------------------
    # -----  Test config  -----
    # -------------------------
    test_partial_decoder = False
    print_architecture = True
    test_encoder = True
    test_encode_decode = True
    test_translation = True

    # -------------------------------------
    # -----  Test gen.decoder.ignore  -----
    # -------------------------------------
    if test_partial_decoder:
        print_header("test_partial_decoder")
        partial_opts = deepcopy(opts)
        partial_opts.gen.a.ignore = False
        partial_opts.gen.d.ignore = True
        partial_opts.gen.h.ignore = False
        partial_opts.gen.t.ignore = False
        partial_opts.gen.w.ignore = False
        G = get_gen(partial_opts).to(device)
        print("d" in G.decoders)
        print("a" in G.decoders)
        z = torch.randn(batch_size, *latent_space_dims, dtype=torch.float32).to(device)
        v = G.decoders["s"](z)
        print(v.shape)
        print(sum(p.numel() for p in G.decoders.parameters()))

    G = get_gen(opts).to(device)

    # -------------------------------
    # -----  Test Architecture  -----
    # -------------------------------
    if print_architecture:
        print(G)
        # print("DECODERS:", G.decoders)
        # print("ENCODER:", G.encoder)

    # ------------------------------------
    # -----  Test encoder.forward()  -----
    # ------------------------------------
    if test_encoder:
        print_header("test_encoder")
        num_params = get_num_params(G.encoder)
        print("Number of parameters in encoder : {}".format(num_params))
        encoded = G.encode(image)
        print("Latent space dims {}".format(tuple(encoded.shape)[1:]))

    # -------------------------------------------------------
    # -----  Test encode then decode with all decoders  -----
    # -------------------------------------------------------
    if test_encode_decode:
        print_header("test_encode_decode")
        z = G.encode(image)
        for dec in "adhtw":
            if dec in G.decoders:
                if dec == "t":
                    continue
                if dec == "a":
                    for d in G.decoders[dec]:
                        print(dec, d, G.decoders[dec][d](z).shape)
                else:
                    print(dec, G.decoders[dec](z).shape)

    # --------------------------
    # -----  Test painter  -----
    # --------------------------

    print_header("testing painter")

    size = 256
    image = torch.Tensor(batch_size, 3, size, size).uniform_(-1, 1).to(device)
    mask = torch.Tensor(batch_size, 3, size, size).uniform_(0, 1).to(device)
    latent_size = size // (2 ** opts.gen.p.spade_n_up)

    painter = SpadeDecoder(opts).to(device)

    # Sample latent vector
    z = (
        torch.empty(batch_size, opts.gen.p.latent_dim, latent_size, latent_size,)
        .normal_(mean=0, std=1.0)
        .to(device)
    )

    out = painter(z, image * mask)
    out2 = G.painter(z, image * mask)
    assert out.shape == out2.shape
    print("Shape of painter output: ", out.shape)
