import sys

sys.path.append("..")

from omnigan.trainer import Trainer
from omnigan.utils import load_opts
from run import print_header

if __name__ == "__main__":
    opts = load_opts("../shared/defaults.yml")
    trainer = Trainer(opts, verbose=1)

    test_setup = True
    test_updage_g_representation = True

    if test_setup:
        print_header("test_setup")
        trainer.setup()

    if test_updage_g_representation:
        print_header("test_updage_g_representation")
        if not trainer.is_setup:
            trainer.setup()
        multi_batch_tuple = next(iter(trainer.train_loaders))
        domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
        trainer.update_g_representation(domain_batch)
