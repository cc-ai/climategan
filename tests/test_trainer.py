import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from omnigan.trainer import Trainer
from omnigan.utils import load_test_opts

# from omnigan.tutils import freeze
from run import print_header

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/trainer/local_tests.yaml")
args = parser.parse_args()
root = Path(__file__).parent.parent
opts = load_test_opts(args.config)


if __name__ == "__main__":
    # ------------------------
    # -----  Test Setup  -----
    # ------------------------
    trainer = Trainer(opts, verbose=1)
    trainer.setup()
    multi_batch_tuple = next(iter(trainer.train_loaders))
    multi_domain_batch = {
        batch["domain"][0]: trainer.batch_to_device(batch)
        for batch in multi_batch_tuple
    }
    # -------------------------
    # -----  Test Config  -----
    # -------------------------
    test_setup = True
    test_get_representation_loss = True
    test_get_translation_loss = True
    test_get_classifier_loss = True
    test_update_g = True
    test_update_d = False
    test_full_step = True

    # ----------------------------------
    # -----  Test trainer.setup()  -----
    # ----------------------------------
    if test_setup:
        print_header("test_setup")
        trainer.setup()

    # ----------------------------------------------------
    # -----  Test trainer.get_masker_loss()  -----
    # ----------------------------------------------------
    if test_get_representation_loss:
        print_header("test_get_masker_loss")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        loss = trainer.get_masker_loss(multi_domain_batch)
        print("Loss {}".format(loss))

    # -------------------------------------------------
    # -----  Test trainer.get_translation_loss()  -----
    # -------------------------------------------------

    # if test_get_translation_loss:
    #     print_header("test_get_translation_loss")
    #     if not trainer.is_setup:
    #         print("Setting up")
    #         trainer.setup()

    #     loss = trainer.get_translation_loss(multi_domain_batch)
    #     print("Loss {}".format(loss))

    # ------------------------------------------------
    # -----  Test trainer.get_classifier_loss()  -----
    # ------------------------------------------------
    if test_get_classifier_loss:
        print_header("test classifier loss")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

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

    # ------------------------------------------------
    # -----  Test trainer.update_g depending on  -----
    # -----  representational_training           -----
    # ------------------------------------------------
    if test_update_g:
        print_header("test_update_g")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        trainer.verbose = 0

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

    # -------------------------------------
    # -----  Test trainer.update_d()  -----
    # -------------------------------------
    if test_update_d:
        print_header("test update_d")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        # trainer.losses["D"].verbose = 0
        print("Decoding using G.decoders[decoder][target_domain]")
        print(
            "Printing \n  {} and \n  {}\n".format(
                "Batch {batch_domain} > {decoder}: {source_domain} to real",
                "Batch {batch_domain} > {decoder}: {target_domain} to fake",
            )
        )
        trainer.logger.global_step = 0
        trainer.update_d(multi_domain_batch, 1)
        trainer.losses["D"].flip_prob = 1.0
        trainer.update_d(multi_domain_batch)

    # -----------------------------------
    # -----  Test full update step  -----
    # -----------------------------------
    if test_full_step:
        trainer.logger.global_step = 0
        trainer.verbose = 0
        # trainer.losses["D"].verbose = 0
        print_header("test FULL STEP")
        if not trainer.is_setup:
            print("Setting up")
            trainer.setup()

        encoder_weights = [
            [p.detach().cpu().numpy()[:5] for p in trainer.G.encoder.parameters()]
        ]

        print("First update: extrapolation")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)

        #! Ignoring discrim update since we aren't
        #! yet doing translation
        # print("  - Update d")
        # trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

        trainer.logger.global_step += 1

        print("Second update: gradient step")
        print("  - Update g")
        trainer.update_g(multi_domain_batch)
        # print("  - Update d")
        # trainer.update_d(multi_domain_batch)
        print("  - Update c")
        trainer.update_c(multi_domain_batch)

