from comet_ml import Experiment
from addict import Dict
from time import time
import torch

from omnigan.generator import get_gen
from omnigan.utils import flatten_opts, freeze, domain_to_class
from omnigan.optim import get_optimizer
from omnigan.data import get_all_loaders
from omnigan.discriminator import get_dis
from omnigan.classifier import get_classifier
from omnigan.losses import cross_entropy_2d


class Trainer:
    def __init__(self, opts, comet=None, verbose=0):
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
            self.logger.exp = Experiment()
            self.logger.exp.log_parameters(flatten_opts(opts))

    def batch_to_device(self, b):
        for task, tensor in b.data.items():
            b.data[task] = tensor.to(self.device)
        return b

    def compute_latent_shape(self):
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
        self.losses = {"G": {}, "D": {}, "C": {}}
        self.losses["G"] = {"gan": {}, "cycle": {}}

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
            self.losses["G"]["d"] = lambda x, y: (x + y).mean()

        if "h" in self.opts.tasks:
            self.losses["G"]["h"] = lambda x, y: (x + y).mean()

        if "s" in self.opts.tasks:
            self.losses["G"]["s"] = cross_entropy_2d

        if "w" in self.opts.tasks:
            self.losses["G"]["w"] = lambda x, y: (x + y).mean()

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

        self.losses

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
        assert self.is_setup
        for i, multi_batch_tuple in enumerate(self.train_loaders):
            multi_domain_batch = {
                batch["domain"][0]: batch for batch in multi_batch_tuple
            }
            self.update_g(multi_domain_batch)
            self.update_d(multi_domain_batch)
            self.update_c(multi_domain_batch)

    def train(self):
        assert self.is_setup

        for self.logger.epoch in range(self.opts.train.epochs):
            self.run_epoch()
            self.eval()
            self.save()

    def update_g(self, multi_domain_batch):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

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

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders
        """
        step_loss = 0
        for batch_domain, batch in multi_domain_batch.items():
            self.z = self.G.encoder(batch["data"]["x"])
            # -----------------------------
            # -----  classifier loss  -----
            # -----------------------------
            # TODO: classifier issue @Adrien
            update_loss = self.losses["G"]["classifier"](
                self.z, domain_to_class(batch_domain, self.z.shape[0])
            )
            step_loss += self.opts.train.lambdas.G.classifier * update_loss

            # ---------------------------------------------
            # -----  task-specific regression losses  -----
            # ---------------------------------------------
            for update_task, update_target in batch["data"].items():
                # task t (=translation) will be done in get_translation_loss
                # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                if update_task not in {"t", "a", "x"}:
                    # ? output features classifier
                    prediction = self.G.decoders[update_task](self.z)
                    update_loss = self.losses["G"][update_task](
                        update_target, prediction
                    )
                    self.logger.task_loss[update_task] = update_loss.item()
                    step_loss += self.opts.train.lambdas.G[update_task] * update_loss

                    self.debug("get_representation_loss", locals(), 0)

            # --------------------------------------------------
            # -----  auto-encoding update for translation  -----
            # --------------------------------------------------
            translation_decoder = batch_domain[-1]
            reconstruction = self.G.decoders["t"][translation_decoder](self.z)
            update_loss = self.losses["G"]["auto"]["t"](
                batch["data"]["x"], reconstruction
            )
            self.logger.losses.auto.t = update_loss.item()
            step_loss += self.opts.train.lambdas.G["auto"]["t"] * update_loss
            self.debug("get_representation_loss", locals(), 1)

            # -------------------------------------------------
            # -----  auto-encoding update for adaptation  -----
            # -------------------------------------------------
            adaptation_decoder = batch_domain[0]
            reconstruction = self.G.decoders["a"][adaptation_decoder](self.z)
            update_loss = self.losses["G"]["auto"]["a"](
                batch["data"]["x"], reconstruction
            )
            self.logger.losses.auto.a = update_loss.item()
            step_loss += self.opts.train.lambdas.G.auto.a * update_loss
            self.debug("get_representation_loss", locals(), 2)

        # -----------------------------------------
        # -----  Adversarial adaptation task  -----
        # -----------------------------------------
        # TODO include semantic matching loss
        # ? * Is this really part of the representation phase => yes
        # ? * freeze second pass => yes
        # ? * how to use noisy labels Alex Lamb ICT (we don't have ground truth in the
        # ?   real world so is it better to use noisy, noisy + ICT or no label in this
        # ?   case?)
        if "a" in self.opts.tasks:
            adaptation_tasks = []
            if "rn" in multi_domain_batch and "sn" in multi_domain_batch:
                adaptation_tasks.append(("rn", "sn"))
                adaptation_tasks.append(("sn", "rn"))
            if "rf" in multi_domain_batch and "sf" in multi_domain_batch:
                adaptation_tasks.append(("rf", "sf"))
                adaptation_tasks.append(("sf", "rf"))
            # adaptation_tasks = [("rn", "sn"), ("rf", "sf")]
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
                    step_loss += self.opts.train.lambdas.G.gan.a * update_loss
                    # --------------------------------------
                    # -----  Translation (Cycle) Loss  -----
                    # --------------------------------------
                    update_loss = self.losses["G"]["cycle"]["a"](real, fake, cycle)
                    self.logger.losses.cycle.a[
                        "{} > {}".format(source_domain, target_domain)
                    ] = update_loss.item()
                    step_loss += self.opts.train.lambdas.G.cycle.a * update_loss
        return step_loss

    def get_translation_loss(self, multi_domain_batch):
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
                        local_vars["predictions"][local_vars["update_task"]].shape,
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
