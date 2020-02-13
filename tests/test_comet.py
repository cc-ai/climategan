import argparse
import atexit
import sys
from pathlib import Path

import comet_ml
from addict import Dict

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.trainer import Trainer
from omnigan.utils import get_comet_rest_api_key, load_opts
from run import bcolors, print_header


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_opts(root / args.config, default=root / "shared/defaults.yaml")


def exit_handler(comet_exp):
    def sub_handler():
        print()
        print(
            bcolors.WARNING + "Error in file. Deleting comet experiment" + bcolors.ENDC
        )
        comet_api.delete_experiment(comet_exp.get_key())

    return sub_handler


if __name__ == "__main__":

    opts = opts.copy()

    should_delete = True

    crop_to = 32  # smaller data for faster tests ; -1 for no

    rest_api_key = get_comet_rest_api_key()
    comet_api = comet_ml.api.API()

    if crop_to > 0:
        opts.data.transforms += [
            Dict({"name": "crop", "ignore": False, "height": crop_to, "width": crop_to})
        ]
    comet_exp = comet_ml.Experiment(project_name="omnigan", auto_metric_logging=False)

    atexit.register(exit_handler(comet_exp))

    trainer = Trainer(opts, comet_exp=comet_exp, verbose=0)
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.setup()
    multi_batch_tuple = next(iter(trainer.train_loaders))
    domain_batch = {
        batch["domain"][0]: trainer.batch_to_device(batch)
        for batch in multi_batch_tuple
    }

    trainer.opts.train.log_level = 1

    print_header("test_log_losses")
    trainer.update_g(domain_batch)
    print("update 1")

    trainer.logger.global_step += 1

    trainer.update_g(domain_batch)
    print("update 2")
    trainer.logger.global_step += 1

    trainer.update_g(domain_batch)
    print("update 3")
    trainer.logger.global_step += 1

    print("Shifting to full log")
    trainer.opts.train.log_level = 2

    trainer.update_g(domain_batch)
    print("update 4")
    trainer.logger.global_step += 1

    trainer.update_g(domain_batch)
    print("update 5")
    trainer.logger.global_step += 1

    trainer.update_g(domain_batch)
    print("update 5")

    trainer.exp.end()

    if should_delete:
        comet_api.delete_experiment(trainer.exp.get_key())
        print(
            "{}Successfully deleted your comet exp. Edit should_delete not to{}".format(
                bcolors.OKBLUE, bcolors.ENDC
            )
        )
    else:
        print(
            "\n{}Your comet experiment was not deleted.".format(bcolors.WARNING),
            "set should_delete to True not to pollute your workspace{}".format(
                bcolors.ENDC
            ),
        )
