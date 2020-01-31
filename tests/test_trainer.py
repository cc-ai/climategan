import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from addict import Dict
from omnigan.trainer import Trainer
from omnigan.utils import load_opts, freeze
from run import print_header, opts

if __name__ == "__main__":
    opts = opts.copy()
    test_setup = False
    test_get_representation_loss = False
    test_get_translation_loss = False
    test_get_classifier_loss = False
    test_update_g = False
    test_update_d = False
    test_full_step = True
    crop_to = 32  # smaller data for faster tests ; -1 for no

    root = Path(__file__).parent.parent
    opts = load_opts(
        root / "config/local_tests.yaml", default=root / "shared/defaults.yml"
    )
    if crop_to > 0:
        opts.data.transforms += [
            Dict({"name": "crop", "ignore": False, "height": crop_to, "width": crop_to})
        ]

    trainer = Trainer(opts, verbose=1)

    if test_setup:
        print_header("test_setup")
        trainer.setup()

    if test_get_representation_loss:
        print_header("test_get_representation_loss")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()
        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        loss = trainer.get_representation_loss(multi_domain_batch)
        print("Loss {}".format(loss.item()))

    if test_get_translation_loss:
        print_header("test_get_translation_loss")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        loss = trainer.get_translation_loss(multi_domain_batch)
        print("Loss {}".format(loss.item()))

    if test_get_classifier_loss:
        print_header("test classifier loss")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        trainer.opts.classifier.loss = "l1"
        trainer.setup()
        loss = trainer.get_classifier_loss(multi_domain_batch)
        print("Loss {}".format(loss.item()))
        trainer.opts.classifier.loss = "l2"
        trainer.setup()
        loss = trainer.get_classifier_loss(multi_domain_batch)
        print("Loss {}".format(loss.item()))
        trainer.opts.classifier.loss = "cross_entropy"
        trainer.setup()
        loss = trainer.get_classifier_loss(multi_domain_batch)
        print("Loss {}".format(loss.item()))

    if test_update_g:

        print_header("test_update_g")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        trainer.verbose = 0

        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        # Using repr_tr and step < repr_step and step % 2 == 0
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 0
        print(True, 100, 0, "Using repr_tr and step < repr_step and step % 2 == 0")
        trainer.update_g(multi_domain_batch, 1)
        print()

        # Using repr_tr and step < repr_step and step % 2 == 1
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 1
        print(True, 100, 1, "Using repr_tr and step < repr_step and step % 2 == 1")
        trainer.update_g(multi_domain_batch, 1)
        print()

        # Using repr_tr and step > repr_step
        trainer.opts.train.representational_training = True
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 200
        print(True, 100, 200, "Using repr_tr and step > repr_step")
        trainer.update_g(multi_domain_batch, 1)
        print()

        # Not Using repr_tr and step < repr_step and step % 2 == 0
        trainer.opts.train.representational_training = False
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 200
        print(
            False, 100, 200, "Not Using repr_tr and step < repr_step and step % 2 == 0"
        )
        trainer.update_g(multi_domain_batch, 1)
        print()

        # Not Using repr_tr and step > repr_step and step % 2 == 1
        trainer.opts.train.representational_training = False
        trainer.opts.train.representation_steps = 100
        trainer.logger.global_step = 201
        print(
            False, 100, 201, "Not Using repr_tr and step > repr_step and step % 2 == 1"
        )
        trainer.update_g(multi_domain_batch, 1)
        print()

    if test_update_d:
        print_header("test update_d")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()
        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
        print("Decoding using G.decoders[decoder][target_domain]")
        print(
            "Printing \n  {} and \n  {}\n".format(
                "Batch {batch_domain} > {decoder}: {source_domain} to real",
                "Batch {batch_domain} > {decoder}: {target_domain} to fake",
            )
        )

        trainer.update_d(multi_domain_batch, 1)
        trainer.losses["D"].flip_prob = 1.0
        trainer.update_d(multi_domain_batch)

    if test_full_step:
        trainer.verbose = 0
        print_header("test FULL STEP")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        encoder_weights = [
            [p.detach().numpy()[0] for p in trainer.G.encoder.parameters()]
        ]
        multi_batch_tuple = next(iter(trainer.train_loaders))
        multi_domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}

        print("First update: extrapolation")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)
        print("  - Update d")
        trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

        trainer.logger.global_step += 1

        print("Second update: gradient step")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)
        print("  - Update d")
        trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

        print("Freezing encoder")
        freeze(trainer.G.encoder)
        trainer.representation_is_frozen = True
        encoder_weights += [[p.numpy()[0] for p in trainer.G.encoder.parameters()]]
        trainer.logger.global_step += 1

        print("Third update: extrapolation")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)
        print("  - Update d")
        trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

        trainer.logger.global_step += 1

        print("Fourth update: gradient step")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)
        print("  - Update d")
        trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

        encoder_weights += [[p.numpy()[0] for p in trainer.G.encoder.parameters()]]

        # ? triggers segmentation fault for some unknown reason
        # # encoder was updated
        # assert all(
        #     [
        #       (p0 != p1).all()
        #       for p0, p1 in zip(encoder_weights[0], encoder_weights[1])
        #     ]
        # )
        # # encoder was not updated
        # assert all(
        #     [
        #       (p1 == p2).all()
        #       for p1, p2 in zip(encoder_weights[1], encoder_weights[2])
        #     ]
        # )
