from addict import Dict
from .models.models import create_model
from numpy import inf

from pathlib import Path

checkpoints = Path(__file__).parent / "checkpoints"

opt = Dict(
    {
        "batchSize": 1,
        "loadSize": 286,
        "fineSize": 256,
        "input_nc": 3,
        "output_nc": 3,
        "ngf": 64,
        "ndf": 64,
        "which_model_netG": "unet_256",
        "gpu_ids": [0, 1],
        "name": "pretrained",
        "model": "pix2pix",
        "nThreads": 2,
        "checkpoints_dir": str(checkpoints),
        "norm": "instance",
        "serial_batches": False,
        "display_winsize": 256,
        "display_id": 1,
        "identity": 0.0,
        "use_dropout": False,
        "max_dataset_size": inf,
        "display_freq": 100,
        "print_freq": 100,
        "save_latest_freq": 5000,
        "save_epoch_freq": 5,
        "continue_train": False,
        "phase": "train",
        "which_epoch": "latest",
        "niter": 100,
        "niter_decay": 100,
        "beta1": 0.5,
        "lr": 0.0002,
        "no_lsgan": False,
        "lambda_A": 10.0,
        "lambda_B": 10.0,
        "pool_size": 50,
        "no_html": False,
        "no_flip": False,
        "isTrain": True,
    }
)


def get_mega_model():
    # ! data transforms should fit MegaDepth's expected values
    model = create_model(opt)
    model.switch_to_eval()
    return model.netG
