import numpy as np
import torch
from addict import Dict
import sys

sys.path.append("..")
from omnigan.trainer import Trainer
from omnigan.utils import load_opts

if __name__ == "__main__":
    opts = load_opts("../shared/defaults.yml")

    trainer = Trainer(opts, verbose=1)
    trainer.setup()
