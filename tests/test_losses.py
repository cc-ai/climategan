import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.data import get_all_loaders
from omnigan.generator import get_gen
from omnigan.losses import (
    PixelCrossEntropy,
    BinaryCrossEntropy,
    entropy_loss,
    MSELoss,
    prob_2_entropy,
    L1Loss,
    ADVENTSegLoss,
    ADVENTAdversarialLoss,
    get_fc_discriminator
)
from omnigan.utils import load_test_opts
from run import print_header

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

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
    prediction = G.decoders["s"](z)

    print("size of prediction: ", prediction.squeeze().size())
    print("size of target: ", batch["data"]["s"].long().squeeze().size())
    # -----------------------------------
    # -----  Test cross_entropy_2d  -----
    # -----------------------------------
    print_header("test_crossentroy_2d......")
    pce = PixelCrossEntropy()
    print(pce(prediction.squeeze(), batch["data"]["s"].long().squeeze().to(device)))
    # ! error how to infer from cropped data: input: 224 output: 256??

    # -----------------------------------
    # -----  Test BinaryCrossEntropy  ---
    # -----------------------------------
    print_header("test_BinaryCrossEntropy......")
    bce = BinaryCrossEntropy()
    pre = torch.nn.Sigmoid()(prediction.squeeze()[:, 0, : , :])
    print(bce(pre, batch["data"]["s"].float().squeeze().to(device)))

    # -----------------------------------
    # -----  Test entropy_loss  ----
    # -----------------------------------
    print_header("test_entropy_loss......")
    print(entropy_loss(prediction.squeeze()))

    # -----------------------------------
    # -----  Test MSELoss  ----
    # -----------------------------------
    print_header("test_MSELoss......")
    mse = MSELoss()
    print(mse(prediction.squeeze()[:, 0, : , :], batch["data"]["s"].float().squeeze().to(device)))

    # -----------------------------------
    # -----  Test L1Loss  ----
    # -----------------------------------
    print_header("test_L1Loss......")
    l1l = L1Loss()
    print(l1l(prediction.squeeze()[:, 0, : , :], batch["data"]["s"].float().squeeze().to(device)))

    # -----------------------------------
    # -----  Test prob_2_entropy  ----
    # -----------------------------------
    print_header("test_prob_2_entropy......")
    pre = torch.nn.Sigmoid()(prediction.squeeze())
    print(prob_2_entropy(pre))

    # -----------------------------------
    # -----  Test ADVENTsegLoss  ----
    # -----------------------------------
    print_header("test_ADVENTsegLoss......")
    ADVseg = ADVENTSegLoss(opts)
    print(ADVseg(prediction.squeeze(), prediction.squeeze(), batch["data"]["s"].squeeze()))

    # -----------------------------------
    # -----  Test ADVENTAdversarialLoss  ----
    # -----------------------------------
    print_header("test_ADVENTAdversarialLoss......")
    d_aux = get_fc_discriminator(num_classes=19)
    d_main = get_fc_discriminator(num_classes=19)
    d_aux.to(device)
    d_main.to(device)
    ADVadversal = ADVENTAdversarialLoss(opts, d_aux, d_main)
    domain_label = 1
    print(ADVadversal(prediction.squeeze(), prediction.squeeze(), domain_label))
