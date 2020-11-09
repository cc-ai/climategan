import atexit
from argparse import ArgumentParser

from comet_ml import Experiment
from comet_ml.api import API
import torch

import omnigan


def exit_handler(exp, api):
    def sub_handler():
        print()
        print("Deleting comet experiment")
        api.delete_experiment(exp.get_key())

    return sub_handler


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--no_delete", action="store_true", default=False)
    args = parser.parse_args()

    opts = omnigan.utils.load_opts()
    opts.data.check_samples = False
    opts.train.fid.n_images = 5
    opts.comet.display_size = 5
    opts.tasks = ["m", "s", "d"]
    opts.domains = ["r", "s"]
    opts.data.loaders.num_workers = 4
    opts.data.loaders.batch_size = 3
    opts.data.max_samples = 9
    opts.train.epochs = 2

    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=None,)
    trainer.setup()
    trainer.train()

    del trainer
    torch.cuda.empty_cache()

    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=Experiment())
    trainer.exp.log_parameter("is_functional_test", True)
    if not args.no_delete:
        from omnigan.utils import get_comet_rest_api_key

        rest_api_key = get_comet_rest_api_key()
        api = API(api_key=rest_api_key)
        atexit.register(exit_handler(trainer.exp, api))

    trainer.setup()
    trainer.train()
