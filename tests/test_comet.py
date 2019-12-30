import sys
import comet_ml

sys.path.append("..")

from omnigan.trainer import Trainer
from omnigan.utils import load_opts, get_comet_rest_api_key
from run import print_header, bcolors

if __name__ == "__main__":

    rest_api_key = get_comet_rest_api_key()
    comet_api = comet_ml.api.API()

    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")
    trainer = Trainer(opts, comet=True, verbose=0)
    trainer.exp.log_parameter("is_functional_test", True)
    trainer.setup()
    multi_batch_tuple = next(iter(trainer.train_loaders))
    domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

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

    should_delete = False
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
