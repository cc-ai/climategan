"""Main component: the trainer handles everything:
    * initializations
    * training
    * saving
"""
from comet_ml import Experiment
from time import time
import torch
from addict import Dict
from pathlib import Path

from omnigan.classifier import get_classifier
from omnigan.data import get_all_loaders
from omnigan.discriminator import get_dis
from omnigan.generator import get_gen
from omnigan.losses import (
    CrossEntropy,
    PixelCrossEntropy,
    L1Loss,
    MSELoss,
    GANLoss,
)
from omnigan.optim import get_optimizer
from omnigan.mega_depth import get_mega_model
from omnigan.utils import flatten_opts
from omnigan.tutils import (
    domains_to_class_tensor,
    fake_batch,
    fake_domains_to_class_tensor,
    freeze,
    save_batch,
    slice_batch,
    shuffle_batch_tuple,
    get_conditioning_tensor,
)


class Trainer:
    """Main trainer class
    """
    def __init__(self, opts, comet_exp=None, verbose=0):
        """Trainer class to gather various model training procedures
        such as training evaluating saving and logging

        init:
        * creates an addict.Dict logger
        * creates logger.exp as a comet_exp experiment if `comet` arg is True
        * sets the device (1 GPU or CPU)

        Args:
            opts (addict.Dict): options to configure the trainer, the data, the models
            comet (bool, optional): whether to log the trainer with comet.ml.
                                    Defaults to False.
            verbose (int, optional): printing level to debug. Defaults to 0.
        """
        super().__init__()

        self.opts = opts
        self.verbose = verbose
        self.logger = Dict()
        self.logger.lr.g = opts.gen.opt.lr
        self.logger.lr.d = opts.dis.opt.lr
        self.logger.epoch = 0
        self.loaders = None

        self.is_setup = False
        self.representation_is_frozen = False
        self.translation_map = {"r": "s", "s": "r", "f": "n", "n": "f"}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.exp = None
        if isinstance(comet_exp, Experiment):
            self.exp = comet_exp

    def log_losses(self, model_to_update="G"):
        """Logs metrics on comet.ml

        Args:
            model_to_update (str, optional): One of "G", "D" or "C". Defaults to "G".
        """
        if self.opts.train.log_level < 1:
            return

        if self.exp is None:
            return

        assert model_to_update in {
            "G",
            "D",
            "C",
        }, "unknown model to log losses {}".format(model_to_update)

        losses = self.logger.losses.copy()
        if self.opts.train.log_level == 1:
            # Only log aggregated losses: delete other keys in losses
            for k in self.logger.losses:
                if k not in {"representation", "generator", "translation"}:
                    del losses[k]
        # convert losses into a single-level dictionnary
        losses = flatten_opts(losses)
        self.exp.log_metrics(
            losses, prefix=model_to_update, step=self.logger.global_step
        )

    def batch_to_device(self, b):
        """sends the data in b to self.device

        Args:
            b (dict): the batch dictionnay

        Returns:
            dict: the batch dictionnary with its "data" field sent to self.device
        """
        for task, tensor in b["data"].items():
            b["data"][task] = tensor.to(self.device)
        return b

    def compute_latent_shape(self):
        """Compute the latent shape, i.e. the Encoder's output shape,
        from a batch.

        Raises:
            ValueError: If no loader, the latent_shape cannot be inferred

        Returns:
            tuple: (c, h, w)
        """
        b = None
        for mode in self.loaders:
            for domain in self.loaders[mode]:
                b = Dict(next(iter(self.loaders[mode][domain])))
                break
        if b is None:
            raise ValueError("No batch found to compute_latent_shape")
        b = self.batch_to_device(b)
        z = self.G.encode(b.data.x)
        return z.shape[1:]

    def set_losses(self):
        """Sets the loss functions to be used by G, D and C, as specified
        in the opts and losses.py

        self.losses = {
            "G": {
                "gan": {"a": ..., "t": ...},
                "cycle": {"a": ..., "t": ...}
                "auto": {"a": ..., "t": ...}
                "tasks": {"h": ..., "d": ..., "s": ..., etc.}
            },
            "D": GANLoss,
            "C": ...
        }
        """

        self.losses = {"G": {"a": {}, "t": {}, "tasks": {}}, "D": {}, "C": {}}
        # ------------------------------
        # -----  Generator Losses  -----
        # ------------------------------

        # translation losses
        if "a" in self.opts.tasks:
            self.losses["G"]["a"]["gan"] = GANLoss()
            self.losses["G"]["a"]["cycle"] = MSELoss()
            self.losses["G"]["a"]["auto"] = MSELoss()

            # ? add sm and dm losses too as in "t"

        if "t" in self.opts.tasks:
            self.losses["G"]["t"]["gan"] = GANLoss()
            self.losses["G"]["t"]["cycle"] = MSELoss()
            self.losses["G"]["t"]["auto"] = MSELoss()
            self.losses["G"]["t"]["sm"] = PixelCrossEntropy()
            self.losses["G"]["t"]["dm"] = MSELoss()

        # task losses
        # ? * add discriminator and gan loss to these task when no ground truth
        # ?   instead of noisy label
        if "d" in self.opts.tasks:
            self.losses["G"]["tasks"]["d"] = MSELoss()

        if "h" in self.opts.tasks:
            self.losses["G"]["tasks"]["h"] = MSELoss()

        if "s" in self.opts.tasks:
            self.losses["G"]["tasks"]["s"] = CrossEntropy()

        if "w" in self.opts.tasks:
            self.losses["G"]["tasks"]["w"] = lambda x, y: (x + y).mean()

        # undistinguishable features loss
        # TODO setup a get_losses func to assign the right loss according to the yaml
        if self.opts.classifier.loss == "l1":
            loss_classifier = L1Loss()
        elif self.opts.classifier.loss == "l2":
            loss_classifier = MSELoss()
        else:
            loss_classifier = CrossEntropy()

        self.losses["G"]["classifier"] = loss_classifier
        # -------------------------------
        # -----  Classifier Losses  -----
        # -------------------------------
        self.losses["C"] = loss_classifier
        # ----------------------------------
        # -----  Discriminator Losses  -----
        # ----------------------------------
        self.losses["D"] = GANLoss(
            soft_shift=self.opts.dis.soft_shift,
            flip_prob=self.opts.dis.flip_prob,
            verbose=self.verbose,
        )

    def setup(self):
        """Prepare the trainer before it can be used to train the models:
            * initialize G and D
            * compute latent space dims and create classifier accordingly
            * creates 3 optimizers
        """
        self.logger.global_step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.loaders = get_all_loaders(self.opts)

        self.G = get_gen(self.opts, verbose=self.verbose).to(self.device)
        self.latent_shape = self.compute_latent_shape()
        self.output_size = self.latent_shape[0] * 2 ** self.opts.gen.t.spade_n_up
        self.G.set_translation_decoder(self.latent_shape, self.device)
        self.D = get_dis(self.opts, verbose=self.verbose).to(self.device)
        self.C = get_classifier(self.opts, self.latent_shape, verbose=self.verbose).to(
            self.device
        )
        self.P = {"s": get_mega_model()}  # P => pseudo labeling models

        self.g_opt, self.g_scheduler = get_optimizer(self.G, self.opts.gen.opt)
        self.d_opt, self.d_scheduler = get_optimizer(self.D, self.opts.dis.opt)
        self.c_opt, self.c_scheduler = get_optimizer(self.C, self.opts.classifier.opt)

        self.set_losses()

        if self.verbose > 0:
            for mode, mode_dict in self.loaders.items():
                for domain, domain_loader in mode_dict.items():
                    print(
                        "Loader {} {} : {}".format(
                            mode, domain, len(domain_loader.dataset)
                        )
                    )

        self.is_setup = True

    def g_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.gen.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.g_opt.extrapolation()
        else:
            self.g_opt.step()

    def d_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.dis.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.d_opt.extrapolation()
        else:
            self.d_opt.step()

    def c_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.classifier.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.c_opt.extrapolation()
        else:
            self.c_opt.step()

    @property
    def train_loaders(self):
        """Get a zip of all training loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["train"].values()))

    def update_learning_rates(self):
        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()
        if self.c_scheduler is not None:
            self.c_scheduler.step()

    @property
    def val_loaders(self):
        """Get a zip of all validation loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["val"].values()))

    def run_epoch(self):
        """Runs an epoch:
        * checks trainer is setup
        * gets a tuple of batches per domain
        * sends batches to device
        * updates sequentially G, D, C
        """
        assert self.is_setup
        for i, multi_batch_tuple in enumerate(self.train_loaders):
            # create a dictionnay (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device
            print(
                "\rEpoch {} batch {} step {}".format(
                    self.logger.epoch, i, self.logger.global_step
                )
            )
            multi_batch_tuple = shuffle_batch_tuple(multi_batch_tuple)
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }
            self.update_g(multi_domain_batch)
            self.update_d(multi_domain_batch)
            self.update_c(multi_domain_batch)
            self.logger.global_step += 1
            if self.should_freeze_representation():
                freeze(self.G.encoder)
                # ? Freeze decoders != t for memory management purposes ; faster ?
                self.representation_is_frozen = True
        self.update_learning_rates()

    def train(self):
        """For each epoch:
        * train
        * eval
        * save
        """
        assert self.is_setup

        for self.logger.epoch in range(self.opts.train.epochs):
            self.run_epoch()
            self.eval()
            self.save()

    def should_freeze_representation(self):
        if self.representation_is_frozen:
            return False
        if not self.opts.train.freeze_representation:
            return False
        if not self.opts.train.representational_training:
            return False
        if self.logger.global_step < self.opts.train.representation_steps:
            return False
        return True

    def should_compute_r_loss(self):
        if not self.opts.train.representational_training:
            return True
        if self.logger.global_step < self.opts.train.representation_steps:
            return True
        return False

    def should_compute_t_loss(self):
        if not self.opts.train.representational_training:
            return True
        if self.logger.global_step >= self.opts.train.representation_steps:
            return True
        return False

    def update_g(self, multi_domain_batch, verbose=0):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

        * compute loss
            * if using Sam Lavoie's representational_training:
                * compute either representation_loss or translation_loss
                  depending on the current step vs opts.train.representation_steps
            * otherwise compute both
        * loss.backward()
        * g_opt_step()
            * g_opt.step() or .extrapolation() depending on self.logger.global_step
        * logs losses on comet.ml with self.log_losses(model_to_update="G")

        Args:
            multi_domain_batch (dict): dictionnary of domain batches
        """
        self.g_opt.zero_grad()
        r_loss = t_loss = None

        if self.should_compute_r_loss():
            r_loss = self.get_representation_loss(multi_domain_batch)

        if self.should_compute_t_loss():
            t_loss = self.get_translation_loss(multi_domain_batch)

        assert any(l is not None for l in [r_loss, t_loss]), "Both losses are None"

        g_loss = 0
        if r_loss is not None:
            g_loss += r_loss
            if verbose > 0:
                print("adding r_loss {} to g_loss".format(r_loss))
            self.logger.losses.representation = r_loss.item()
        if t_loss is not None:
            g_loss += t_loss
            if verbose > 0:
                print("adding t_loss {} to g_loss".format(t_loss))
            self.logger.losses.translation = t_loss.item()
        if verbose > 0:
            print("g_loss is {}".format(g_loss))
        self.logger.losses.generator = g_loss.item()
        g_loss.backward()
        self.g_opt_step()
        self.log_losses(model_to_update="G")

    def get_representation_loss(self, multi_domain_batch):
        """Only update the representation part of the model, meaning everything
        but the translation part

        * for each batch in available domains:
            * compute latent classifier loss with fake labels(1)
            * compute task-specific losses (2)
            * compute the adaptation and translation decoders' auto-encoding losses (3)
            * compute the adaptation decoder's translation losses (GAN and Cycle) (4)

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        step_loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        # ? should we add all domains to the loss (.backward() and .step() after this
        # ? loop) or update the networks for each domain sequentially
        # ? (.backward() and .step() n times)?
        for batch_domain, batch in multi_domain_batch.items():
            x = batch["data"]["x"]
            self.z = self.G.encode(x)
            # ---------------------------------
            # -----  classifier loss (1)  -----
            # ---------------------------------
            # Forward pass through classifier, output : (batch_size, 4)
            output_classifier = self.C(self.z)
            # Cross entropy loss (with sigmoid) with fake labels to fool C
            update_loss = self.losses["G"]["classifier"](
                output_classifier,
                fake_domains_to_class_tensor(batch["domain"], one_hot),
            )
            step_loss += lambdas.G.classifier * update_loss
            # -------------------------------------------------
            # -----  task-specific regression losses (2)  -----
            # -------------------------------------------------
            task_tensors = {}
            for update_task, update_target in batch["data"].items():
                # task t (=translation) will be done in get_translation_loss
                # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                if update_task not in {"t", "a", "x"}:
                    # ? output features classifier
                    prediction = self.G.decoders[update_task](self.z)
                    task_tensors[update_task] = prediction
                    update_loss = self.losses["G"]["tasks"][update_task](
                        prediction, update_target
                    )
                    step_loss += lambdas.G[update_task] * update_loss
                    self.logger.losses.task_loss[update_task][
                        batch_domain
                    ] = update_loss.item()

                    self.debug("get_representation_loss", locals(), 0)
            # ------------------------------------------------------
            # -----  auto-encoding update for translation (3)  -----
            # ------------------------------------------------------
            translation_decoder = batch_domain[-1]

            cond = None
            if self.opts.gen.t.use_spade:
                cond = get_conditioning_tensor(x, task_tensors)

            reconstruction = self.G.decoders["t"][translation_decoder](self.z, cond)
            update_loss = self.losses["G"]["t"]["auto"](x, reconstruction)
            step_loss += lambdas.G.t.auto * update_loss
            self.logger.losses.t.auto[batch_domain] = update_loss.item()
            self.debug("get_representation_loss", locals(), 1)

            # -----------------------------------------------------
            # -----  auto-encoding update for adaptation (3)  -----
            # -----------------------------------------------------
            adaptation_decoder = batch_domain[0]
            reconstruction = self.G.decoders["a"][adaptation_decoder](self.z)
            update_loss = self.losses["G"]["a"]["auto"](x, reconstruction)
            step_loss += lambdas.G.a.auto * update_loss
            self.logger.losses.a.auto[batch_domain] = update_loss.item()
            self.debug("get_representation_loss", locals(), 2)

        # ---------------------------------------------
        # -----  Adaptation translation task (4)  -----
        # ---------------------------------------------
        # TODO include semantic matching loss
        # ? * Is this really part of the representation phase => yes
        # ? * freeze second pass => yes
        # ? * how to use noisy labels Alex Lamb ICT (we don't have ground truth in the
        # ?   real world so is it better to use noisy, noisy + ICT or no label in this
        # ?   case?)

        # only do this if adaptation is specified in opts
        if "a" in self.opts.tasks:
            adaptation_tasks = []
            if "rn" in multi_domain_batch and "sn" in multi_domain_batch:
                adaptation_tasks.append(("rn", "sn"))
                adaptation_tasks.append(("sn", "rn"))
            if "rf" in multi_domain_batch and "sf" in multi_domain_batch:
                adaptation_tasks.append(("rf", "sf"))
                adaptation_tasks.append(("sf", "rf"))
            # adaptation_tasks = all adaptation possible real to sim and sim to real,
            # flooded or not
            assert len(adaptation_tasks) > 0

            for source_domain, target_domain in adaptation_tasks:

                real = multi_domain_batch[source_domain]["data"]["x"]
                z_real = self.G.encode(real)
                fake = self.G.decoders["a"][target_domain[0]](z_real)
                z_fake = self.G.encode(fake)
                cycle = self.G.decoders["a"][source_domain[0]](z_fake)

                self.debug("get_representation_loss", locals(), 3)

                d_fake = self.D["a"][target_domain[0]](fake)
                d_cycle = self.D["a"][source_domain[0]](cycle)

                # ----------------------
                # -----  GAN Loss  -----
                # ----------------------
                update_loss = self.losses["G"]["a"]["gan"](d_fake, True)
                step_loss += lambdas.G.a.gan * update_loss
                self.logger.losses.a.gan[
                    "{} > {}".format(source_domain, target_domain)
                ] = update_loss.item()
                # ? compute GAN loss on cycle reconstruction?
                update_loss = self.losses["G"]["a"]["gan"](d_cycle, True)
                step_loss += lambdas.G.a.gan * update_loss
                self.logger.losses.a.gan[
                    "{} > {}".format(source_domain, target_domain)
                ] += update_loss.item()
                # --------------------------------------
                # -----  Translation (Cycle) Loss  -----
                # --------------------------------------
                update_loss = self.losses["G"]["a"]["cycle"](real, cycle)
                step_loss += lambdas.G.a.cycle * update_loss
                self.logger.losses.a.cycle[
                    "{} > {}".format(source_domain, target_domain)
                ] = update_loss.item()
        return step_loss

    def get_translation_loss(self, multi_domain_batch):
        """Computes the translation loss when flooding/deflooding images

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        step_loss = 0
        self.g_opt.zero_grad()
        lambdas = self.opts.train.lambdas

        translation_tasks = []
        if "rn" in multi_domain_batch and "rf" in multi_domain_batch:
            translation_tasks.append(("rn", "rf"))
            translation_tasks.append(("rf", "rn"))
        if "sn" in multi_domain_batch and "sf" in multi_domain_batch:
            translation_tasks.append(("sn", "sf"))
            translation_tasks.append(("sf", "sn"))

        # ? * same loop as in representation task but could be different
        # ?   when there's the spade components
        for source_domain, target_domain in translation_tasks:

            batch = multi_domain_batch[source_domain]
            real = batch["data"]["x"]
            real_z = self.G.encode(real)
            fake = self.G.translate_batch(batch, target_domain[1], z=real_z)
            fake_z = self.G.encode(fake)
            cycle_batch = fake_batch(batch, fake)
            cycle = self.G.translate_batch(cycle_batch, source_domain[1], z=fake_z)

            d_fake = self.D["t"][target_domain[1]](fake)
            d_cycle = self.D["t"][source_domain[1]](cycle)

            # ----------------------
            # -----  GAN Loss  -----
            # ----------------------
            update_loss = self.losses["G"]["t"]["gan"](d_fake, True)
            step_loss += lambdas.G.t.gan * update_loss
            self.logger.losses.t.gan[
                "{} > {}".format(source_domain, target_domain)
            ] = update_loss.item()
            # ? compute GAN loss on cycle reconstruction?
            update_loss = self.losses["G"]["t"]["gan"](d_cycle, True)
            step_loss += lambdas.G.t.gan * update_loss
            self.logger.losses.t.gan[
                "{} > {}".format(source_domain, target_domain)
            ] += update_loss.item()
            # --------------------------------------
            # -----  Translation (Cycle) Loss  -----
            # --------------------------------------
            update_loss = self.losses["G"]["t"]["cycle"](real, cycle)
            step_loss += lambdas.G.t.cycle * update_loss
            self.logger.losses.t.cycle[
                "{} > {}".format(source_domain, target_domain)
            ] = update_loss.item()

            # -----------------------------
            # -----                   -----
            # -----  Matching Losses  -----
            # -----                   -----
            # -----------------------------

            # ------------------------------------
            # -----  Semantic-matching loss  -----
            # ------------------------------------
            fake_s = self.G.decoders["s"](fake_z).detach()
            real_s_labels = torch.argmax(self.G.decoders["s"](real_z).detach(), 1)
            mask = (
                torch.randint(0, 2, real_s_labels.shape)
                .to(torch.float32)
                .to(self.device)
            )  # TODO : load mask
            update_loss = (
                self.losses["G"]["t"]["sm"](fake_s, real_s_labels) * mask
            ).mean()
            step_loss += lambdas.G.t.sm * update_loss
            self.logger.losses.t.sm[
                "{} > {}".format(source_domain, target_domain)
            ] = update_loss.item()
            # ---------------------------------
            # -----  Depth-matching loss  -----
            # ---------------------------------
            fake_d = self.G.decoders["d"](fake_z).detach()
            real_d = self.G.decoders["d"](real_z).detach()
            mask = (
                torch.randint(0, 2, fake_d.shape).to(torch.float32).to(self.device)
            )  # TODO: load mask
            update_loss = self.losses["G"]["t"]["dm"](fake_d * mask, real_d * mask)
            step_loss += lambdas.G.t.dm * update_loss
            self.logger.losses.t.dm[
                "{} > {}".format(source_domain, target_domain)
            ] = update_loss.item()

        return step_loss

    def update_d(self, multi_domain_batch, verbose=0):
        # ? split representational as in update_g
        # ? repr: domain-adaptation traduction
        self.d_opt.zero_grad()
        d_loss = self.get_d_loss(multi_domain_batch, verbose)
        d_loss.backward()
        self.d_opt_step()

        self.logger.losses.discriminator.total_loss = d_loss.item()
        self.log_losses(model_to_update="D")

    def get_d_loss(self, multi_domain_batch, verbose=0):
        """Compute the discriminators' losses:

        * for each domain-specific batch:
        * encode the image
        * get the conditioning tensor if using spade
        * source domain is the data's domain, sequentially r|s then f|n
        * get the target domain accordingly
        * compute the translated image from the data
        * compute the source domain discriminator's loss on the data
        * compute the target domain discriminator's loss on the translated image

        # ? In this setting, each D[decoder][domain] is updated twice towards
        # real or fake data

        See readme's update d section for details

        Args:
            multi_domain_batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        for batch_domain, batch in multi_domain_batch.items():

            x = batch["data"]["x"]
            z = self.G.encode(x)
            cond = None
            if self.opts.gen.t.use_spade:
                task_tensors = self.G.decode_tasks(z)
                cond = get_conditioning_tensor(x, task_tensors)

            disc_loss = {"a": {"r": 0, "s": 0}, "t": {"f": 0, "n": 0}}

            for i, source_domain in enumerate(batch_domain):
                target_domain = self.translation_map[source_domain]
                decoder = "a" if i == 0 else "t"
                fake = self.G.decoders[decoder][target_domain](z, cond)
                fake_d = self.D[decoder][target_domain](fake)
                real_d = self.D[decoder][source_domain](x)
                fake_loss = self.losses["D"](fake_d, False)
                real_loss = self.losses["D"](real_d, True)
                disc_loss[decoder][target_domain] += fake_loss
                disc_loss[decoder][source_domain] += real_loss

                if verbose > 0:
                    print(f"Batch {batch_domain} > {decoder}: {source_domain} to real ")
                    print(f"Batch {batch_domain} > {decoder}: {target_domain} to fake ")

        self.logger.losses.discriminator.update(
            {dom: {k: v.item() for k, v in d.items()} for dom, d in disc_loss.items()}
        )
        loss = sum(v for d in disc_loss.values() for k, v in d.items())
        return loss

    def update_c(self, multi_domain_batch):
        """
        Update the classifier using normal labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
                the trainer's loaders

        """
        self.c_opt.zero_grad()
        c_loss = self.get_classifier_loss(multi_domain_batch)
        # ? Log policy
        self.logger.losses.classifier = c_loss.item()
        c_loss.backward()
        self.c_opt_step()

    def get_classifier_loss(self, multi_domain_batch):
        """Compute the loss of the domain classifier with real labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas.C
        """
        loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        for batch_domain, batch in multi_domain_batch.items():
            self.z = self.G.encode(batch["data"]["x"])
            # Forward through classifier, output classifier = (batch_size, 4)
            output_classifier = self.C(self.z)
            # Cross entropy loss (with sigmoid)
            update_loss = self.losses["C"](
                output_classifier,
                domains_to_class_tensor(batch["domain"], one_hot).to(self.device),
            )
            loss += update_loss

        return lambdas.C * loss

    def eval(self, num_threads=5, verbose=0):
        counter = {}
        for i, multi_batch_tuple in enumerate(self.val_loaders):
            # create a dictionnay (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }
            # ----------------------------------------------
            # -----  Infer separately for each domain  -----
            # ----------------------------------------------
            for domain, domain_batch in multi_domain_batch.items():
                # Don't infer if domains has enough images
                remaining = self.opts.val.max_log_images - counter.get(domain, 0)
                if remaining <= 0:
                    continue
                if verbose > 0:
                    print("\rInferring batch {} domain {}".format(i, domain), end="")

                translator = "f" if "n" in domain else "n"
                domain_batch = slice_batch(domain_batch, remaining)
                translated = self.G.translate(domain_batch, translator)
                domain_batch["data"]["y"] = translated
                multi_domain_batch[domain] = domain_batch
                counter[domain] = counter.get(domain, 0) + translated.shape[0]

            write_path = Path(self.opts.output_path) / "eval_images"
            step = self.logger.global_step
            save_batch(multi_domain_batch, write_path, step, num_threads)
        if verbose > 0:
            print()

    def save(self):
        pass

    def resume(self):
        pass

    def debug(self, func_name, local_vars, index=None):
        if self.verbose == 0:
            return

        if func_name == "get_representation_loss":
            if index == 0:
                print(
                    "Domain {} Task {} Pred {} Loss {}".format(
                        local_vars["batch_domain"],
                        local_vars["update_task"],
                        local_vars["prediction"].shape,
                        local_vars["update_loss"],
                    )
                )
            if index == 1:
                print(
                    "Translation reconstruction {} Loss {}".format(
                        local_vars["reconstruction"].shape, local_vars["update_loss"]
                    )
                )
            if index == 2:
                print(
                    "Adaptation reconstruction {} Loss {}".format(
                        local_vars["reconstruction"].shape, local_vars["update_loss"]
                    )
                )
            if index == 3:
                print("{}: {}".format("real", local_vars["real"].shape))
                print("{}: {}".format("z_real", local_vars["z_real"].shape))
                print("{}: {}".format("fake", local_vars["fake"].shape))

        if func_name == "get_translation_loss":
            print("{}: {}".format("real", local_vars["real"].shape))
            print("{}: {}".format("z_real", local_vars["z_real"].shape))
            print("{}: {}".format("fake", local_vars["fake"].shape))
            print("{}: {}".format("cycle", local_vars["cycle"].shape))
        # print("-------- end {} --------".format(func_name))
        print()
