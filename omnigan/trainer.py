from comet_ml import Experiment
from addict import Dict
from time import time
import torch

from omnigan.generator import get_gen
from omnigan.utils import flatten_opts, freeze, domains_to_class_tensor
from omnigan.optim import get_optimizer
from omnigan.data import get_all_loaders
from omnigan.discriminator import get_dis
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy_2d


class Trainer:
    def __init__(self, opts, comet=False, verbose=0):
        """Trainer class to gather various model training procedures
        such as training evaluating saving and logging

        init:
        * creates an addict.Dict logger
        * creates logger.exp as a comet experiment if `comet` arg is True
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
        self.loaders = None

        self.is_setup = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.exp = None
        if comet:
            self.logger.exp = Experiment(project_name="OmniGAN")
            self.logger.exp.log_parameters(flatten_opts(opts))

    def log_losses(self):
        # TODO
        pass

    def batch_to_device(self, b):
        """sends the data in b to self.device

        Args:
            b (dict): the batch dictionnay

        Returns:
            dict: the batch dictionnary with its "data" field sent to self.device
        """
        for task, tensor in b.data.items():
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
        z = self.G.encoder(b.data.x)
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
            "D": #TODO specify interface here,
            "C": ...
        }
        """
        self.losses = {"G": {"gan": {}, "cycle": {}, "tasks": {}}, "D": {}, "C": {}}

        # ------------------------------
        # -----  Generator Losses  -----
        # ------------------------------

        # translation losses
        if "a" in self.opts.tasks:
            self.losses["G"]["gan"]["a"] = lambda x, y: (x + y).mean()
            self.losses["G"]["cycle"]["a"] = lambda x, y, z: (x + y + z).mean()

        if "t" in self.opts.tasks:
            self.losses["G"]["gan"]["t"] = lambda x, y: (x + y).mean()
            self.losses["G"]["cycle"]["t"] = lambda x, y, z: (x + y + z).mean()

        # task losses
        # ? * add discriminator and gan loss to these task when no ground truth
        # ?   instead of noisy label
        if "d" in self.opts.tasks:
            self.losses["G"]["tasks"]["d"] = lambda x, y: (x + y).mean()

        if "h" in self.opts.tasks:
            self.losses["G"]["tasks"]["h"] = lambda x, y: (x + y).mean()

        if "s" in self.opts.tasks:
            self.losses["G"]["tasks"]["s"] = cross_entropy_2d

        if "w" in self.opts.tasks:
            self.losses["G"]["tasks"]["w"] = lambda x, y: (x + y).mean()

        # auto-encoder losses
        self.losses["G"]["auto"] = {
            "a": lambda x, y: (x + y).mean(),
            "t": lambda x, y: (x + y).mean(),
        }

        # undistinguishable features loss
        self.losses["G"]["classifier"] = lambda x, y: x.mean() + y.mean()

        # -------------------------------
        # -----  Classifier Losses  -----
        # -------------------------------
        # TODO @Adrien

        # ----------------------------------
        # -----  Discriminator Losses  -----
        # ----------------------------------
        # TODO

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
        self.D = get_dis(self.opts, verbose=self.verbose).to(self.device)
        self.latent_shape = self.compute_latent_shape()
        self.C = get_classifier(self.opts, self.latent_shape, verbose=self.verbose).to(
            self.device
        )

        self.g_opt = get_optimizer(self.G, self.opts.gen.opt)
        self.d_opt = get_optimizer(self.D.models, self.opts.dis.opt)
        self.c_opt = get_optimizer(self.C, self.opts.classifier.opt)

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

    @property
    def train_loaders(self):
        """Get a zip of all training loaders

        Returns:
            generator: zip generator
        """
        return zip(*list(self.loaders["train"].values()))

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
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }
            self.update_g(multi_domain_batch)
            self.update_d(multi_domain_batch)
            self.update_c(multi_domain_batch)

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

    def update_g(self, multi_domain_batch):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

        * compute loss
            * if using Sam Lavoie's representational_training:
                * compute either representation_loss or translation_loss
                  depending on the current step vs opts.train.representation_steps
            * otherwise compute both
        * loss.backward()
        * g_opt_step()
            * g_opt.step() or .extrapolation() depending

        Args:
            multi_domain_batch (dict): dictionnary of domain batches
        """
        self.g_opt.zero_grad()
        r_loss = t_loss = None
        if (
            not self.opts.train.representational_training
            or self.logger.global_step < self.opts.train.representation_steps
        ):
            r_loss = self.get_representation_loss(multi_domain_batch)
        if (
            not self.opts.train.representational_training
            or self.logger.global_step >= self.opts.train.representation_steps
        ):
            t_loss = self.get_translation_loss(multi_domain_batch)

        assert any(l is not None for l in [r_loss, t_loss])

        g_loss = sum(filter(lambda x: x is not None, [r_loss, t_loss]))
        if r_loss is not None:
            self.logger.losses.representation = r_loss.item()
        if t_loss is not None:
            self.logger.losses.translation = t_loss.item()
        self.logger.losses.generator = g_loss.item()
        g_loss.backward()
        self.g_opt_step()

    def get_representation_loss(self, multi_domain_batch):
        """Only update the representation part of the model, meaning everything
        but the translation part

        * for each batch in available domains:
            * compute latent classifier loss (1)
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
        # ? should we add all domains to the loss (.backward() and .step() after this
        # ? loop) or update the networks for each domain sequentially?
        for batch_domain, batch in multi_domain_batch.items():
            self.z = self.G.encoder(batch["data"]["x"])
            # ---------------------------------
            # -----  classifier loss (1)  -----
            # ---------------------------------
            # TODO: classifier issue @Adrien
            update_loss = self.losses["G"]["classifier"](
                self.z, domains_to_class_tensor(batch["domain"])
            )
            step_loss += lambdas.G.classifier * update_loss

            # -------------------------------------------------
            # -----  task-specific regression losses (2)  -----
            # -------------------------------------------------
            for update_task, update_target in batch["data"].items():
                # task t (=translation) will be done in get_translation_loss
                # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                if update_task not in {"t", "a", "x"}:
                    # ? output features classifier
                    prediction = self.G.decoders[update_task](self.z)
                    update_loss = self.losses["G"]["tasks"][update_task](
                        prediction, update_target
                    )
                    self.logger.losses.task_loss[update_task][batch_domain] = update_loss.item()
                    step_loss += lambdas.G.tasks[update_task][batch_domain] * update_loss

                    self.debug("get_representation_loss", locals(), 0)

            # ------------------------------------------------------
            # -----  auto-encoding update for translation (3)  -----
            # ------------------------------------------------------
            translation_decoder = batch_domain[-1]
            reconstruction = self.G.decoders["t"][translation_decoder](self.z)
            update_loss = self.losses["G"]["auto"]["t"](
                batch["data"]["x"], reconstruction
            )
            self.logger.losses.auto.t[batch_domain] = update_loss.item()
            step_loss += lambdas.G["auto"]["t"] * update_loss
            self.debug("get_representation_loss", locals(), 1)

            # -----------------------------------------------------
            # -----  auto-encoding update for adaptation (3)  -----
            # -----------------------------------------------------
            adaptation_decoder = batch_domain[0]
            reconstruction = self.G.decoders["a"][adaptation_decoder](self.z)
            update_loss = self.losses["G"]["auto"]["a"](
                batch["data"]["x"], reconstruction
            )
            self.logger.losses.auto.a[batch_domain] = update_loss.item()
            step_loss += lambdas.G.auto.a * update_loss
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

        # only do this if adaptaion is specified in opts
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

                real_source = multi_domain_batch[source_domain]["data"]["x"]
                real_target = multi_domain_batch[target_domain]["data"]["x"]

                z_real_source = self.G.encoder(real_source)
                z_real_target = self.G.encoder(real_target)

                fake_source = self.G.decoders["a"][source_domain[0]](z_real_target)
                fake_target = self.G.decoders["a"][target_domain[0]](z_real_source)

                z_fake_source = self.G.encoder(fake_source)
                z_fake_target = self.G.encoder(fake_target)

                cycle_source = self.G.decoders["a"][source_domain[0]](z_fake_target)
                cycle_target = self.G.decoders["a"][target_domain[0]](z_fake_source)

                self.debug("get_representation_loss", locals(), 3)

                for real, fake, cycle in [
                    (real_source, fake_source, cycle_source),
                    (real_target, fake_target, cycle_target),
                ]:
                    # ----------------------
                    # -----  GAN Loss  -----
                    # ----------------------
                    update_loss = self.losses["G"]["gan"]["a"](real, fake)
                    self.logger.losses.gan.a[
                        "{} > {}".format(source_domain, target_domain)
                    ] = update_loss.item()
                    step_loss += lambdas.G.gan.a * update_loss
                    # --------------------------------------
                    # -----  Translation (Cycle) Loss  -----
                    # --------------------------------------
                    update_loss = self.losses["G"]["cycle"]["a"](real, fake, cycle)
                    self.logger.losses.cycle.a[
                        "{} > {}".format(source_domain, target_domain)
                    ] = update_loss.item()
                    step_loss += lambdas.G.cycle.a * update_loss
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
        if self.opts.train.freeze_representation:
            freeze(self.G.encoder)
            # ? do I need to also freeze the decoders other than t?

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
            real_source = multi_domain_batch[source_domain]["data"]["x"]
            real_target = multi_domain_batch[target_domain]["data"]["x"]

            z_real_source = self.G.encoder(real_source)
            z_real_target = self.G.encoder(real_target)

            fake_source = self.G.decoders["t"][source_domain[1]](z_real_target)
            fake_target = self.G.decoders["t"][target_domain[1]](z_real_source)

            # ? use this as more training data for classifier
            z_fake_source = self.G.encoder(fake_source)
            z_fake_target = self.G.encoder(fake_target)

            cycle_source = self.G.decoders["t"][source_domain[1]](z_fake_target)
            cycle_target = self.G.decoders["t"][target_domain[1]](z_fake_source)

            self.debug("get_translation_loss", locals())

            for real, fake, cycle in [
                (real_source, fake_source, cycle_source),
                (real_target, fake_target, cycle_target),
            ]:
                # ----------------------
                # -----  GAN Loss  -----
                # ----------------------
                update_loss = self.losses["G"]["gan"]["t"](real, fake)
                self.logger.losses.gan.t[
                    "{} > {}".format(source_domain, target_domain)
                ] = update_loss.item()
                step_loss += self.opts.train.lambdas.G.gan.t * update_loss
                # --------------------------------------
                # -----  Translation (Cycle) Loss  -----
                # --------------------------------------
                update_loss = self.losses["G"]["cycle"]["t"](real, fake, cycle)
                self.logger.losses.cycle.t[
                    "{} > {}".format(source_domain, target_domain)
                ] = update_loss.item()
                step_loss += self.opts.train.lambdas.G.cycle.t * update_loss
        return step_loss

    def update_d(self, batch):
        # ? split representational as in update_g
        # ? repr: domain-adaptation traduction
        # ?
        pass

    def update_c(self, batch):
        pass

    def eval(self):
        pass

    def save(self):
        pass

    def resume(self):
        pass

    def debug(self, func_name, local_vars, index=None):
        if self.verbose == 0:
            return

        # pdb.set_trace()

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
                print("{}: {}".format("real_source", local_vars["real_source"].shape))
                print("{}: {}".format("real_target", local_vars["real_target"].shape))
                print(
                    "{}: {}".format("z_real_source", local_vars["z_real_source"].shape)
                )
                print(
                    "{}: {}".format("z_real_target", local_vars["z_real_target"].shape)
                )
                print("{}: {}".format("fake_source", local_vars["fake_source"].shape))
                print("{}: {}".format("fake_target", local_vars["fake_target"].shape))

        if func_name == "get_translation_loss":
            print("{}: {}".format("real_source", local_vars["real_source"].shape))
            print("{}: {}".format("real_target", local_vars["real_target"].shape))
            print("{}: {}".format("z_real_source", local_vars["z_real_source"].shape))
            print("{}: {}".format("z_real_target", local_vars["z_real_target"].shape))
            print("{}: {}".format("fake_source", local_vars["fake_source"].shape))
            print("{}: {}".format("fake_target", local_vars["fake_target"].shape))
