import sys

sys.path.append("..")
from addict import Dict
from omnigan.trainer import Trainer
from omnigan.utils import load_opts
from run import print_header

if __name__ == "__main__":

    test_setup = True
    test_get_representation_loss = True
    test_get_translation_loss = True
    test_get_classifier_loss = True
    test_update_g = True
    crop_to = 32  # smaller data for faster tests ; -1 for no

    opts = load_opts("../config/local_tests.yaml", default="../shared/defaults.yml")
    if crop_to > 0:
        opts.data.transforms = list(
            map(
                lambda x: Dict({**x, "height": crop_to, "width": crop_to})
                if x["name"] == "crop"
                else x,
                opts.data.transforms,
            )
        )
    trainer = Trainer(opts, verbose=1)

    if test_setup:
        print_header("test_setup")
        trainer.setup()

    if test_get_representation_loss:
        print_header("test_get_representation_loss")
        if not trainer.is_setup:
            trainer.setup()
        multi_batch_tuple = next(iter(trainer.train_loaders))
        domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
        loss = trainer.get_representation_loss(domain_batch)
        print("Loss {}".format(loss.item()))

    if test_get_translation_loss:
        print_header("test_get_translation_loss")
        if not trainer.is_setup:
            trainer.setup()

        multi_batch_tuple = next(iter(trainer.train_loaders))
        domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
        loss = trainer.get_translation_loss(domain_batch)
        print("Loss {}".format(loss.item()))

    if test_get_classifier_loss:
        print_header("test classifier loss")
        if not trainer.is_setup:
            trainer.setup()

        multi_batch_tuple = next(iter(trainer.train_loaders))
        domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
        trainer.opts.classifier.loss = "l1"
        trainer.setup()
        loss = trainer.get_classifier_loss(domain_batch)
        print("Loss {}".format(loss.item()))
        trainer.opts.classifier.loss = "l2"
        trainer.setup()
        loss = trainer.get_classifier_loss(domain_batch)
        print("Loss {}".format(loss.item()))
        trainer.opts.classifier.loss = "cross_entropy"
        trainer.setup()
        loss = trainer.get_classifier_loss(domain_batch)
        print("Loss {}".format(loss.item()))

    if test_update_g:

        print_header("test_update_g")
        if not trainer.is_setup:
            trainer.setup()

        trainer.verbose = 0

        multi_batch_tuple = next(iter(trainer.train_loaders))
        domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        # Using repr_tr and step < repr_step and step % 2 == 0
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 0
        print(True, 100, 0, "Using repr_tr and step < repr_step and step % 2 == 0")
        trainer.update_g(domain_batch, 1)
        print()

        # Using repr_tr and step < repr_step and step % 2 == 1
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 1
        print(True, 100, 1, "Using repr_tr and step < repr_step and step % 2 == 1")
        trainer.update_g(domain_batch, 1)
        print()

        # Using repr_tr and step > repr_step
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 200
        print(True, 100, 200, "Using repr_tr and step > repr_step")
        trainer.update_g(domain_batch, 1)
        print()

        # Not Using repr_tr and step < repr_step and step % 2 == 0
        trainer.opts.train.representational_training = False
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 200
        print(
            False, 100, 200, "Not Using repr_tr and step < repr_step and step % 2 == 0"
        )
        trainer.update_g(domain_batch, 1)
        print()

        # Not Using repr_tr and step > repr_step and step % 2 == 1
        trainer.opts.train.representational_training = False
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 201
        print(
            False, 100, 201, "Not Using repr_tr and step > repr_step and step % 2 == 1"
        )
        trainer.update_g(domain_batch, 1)
        print()
