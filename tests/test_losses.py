import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import get_all_loaders
from omnigan.generator import get_gen
from omnigan.losses import PixelCrossEntropy, NTXentLoss
from omnigan.utils import load_test_opts
from run import print_header
from omnigan.simclr import NetSimCLR


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    opts.data.loaders.batch_size = 2
    opts.data.loaders.num_workers = 2
    opts.data.loaders.shuffle = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_all_loaders(opts)
    batch = next(iter(loaders["train"]["r"]))
    image = torch.randn(opts.data.loaders.batch_size, 3, 32, 32).to(device)
    G = get_gen(opts).to(device)
    z = G.encode(image)

    # -----------------------------------
    # -----  Test cross_entropy_2d  -----
    # -----------------------------------
    print_header("test_crossentroy_2d")
    prediction = G.decoders["s"](z)
    pce = PixelCrossEntropy()
    print(pce(prediction.squeeze(), batch["data"]["s"].long().squeeze().to(device)))
    #! error how to infer from cropped data: input: 224 output: 256??

    # -----------------------------------
    # -------  Test NTXentLoss()  -------
    # -----------------------------------
    # For now, testing the loss on representations directly (eventually, test it on projections)
    print_header("test_ntxent")

    opts.gen.encoder.architecture = "base"
    opts.gen.default.res_dim = 256
    opts.gen.a.ignore = True

    G = get_gen(opts).to(device)
    z = G.encode(image)

    latent_shape = z.shape[1:]
    simclr_model = NetSimCLR(G.encoder, latent_shape, opts.gen.simclr.output_size)

    image1 = torch.randn(opts.data.loaders.batch_size, 3, 32, 32).to(device)
    image2 = torch.randn(opts.data.loaders.batch_size, 3, 32, 32).to(device)
    h1, z1 = simclr_model.forward(image1)
    h2, z2 = simclr_model.forward(image2)

    print("Shape of projections:", z1.shape)
    loss = NTXentLoss(opts.data.loaders.batch_size, 0.5, True)
    print("NTXent loss:", loss(z1, z2).to(device))

    # TODO more test for the losses
