from comet_ml import Experiment
from addict import Dict
from time import time
import torch

from omnigan.generator import get_gen
from omnigan.utils import flatten_opts, freeze
from omnigan.optim import get_optimizer
from omnigan.data import get_all_loaders
from omnigan.discriminator import get_dis
from omnigan.classifier import get_classifier


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
        self.losses = {}

        # translation losses
        if "a" in self.opts.tasks:
            self.losses["a"] = lambda x, y: (x + y).mean()

        if "t" in self.opts.tasks:
            self.losses["t"] = lambda x, y: (x + y).mean()

        # task losses
        # ? add discriminator and gan loss to these task when no ground truth
        # ? instead of noisy label
        if "d" in self.opts.tasks:
            self.losses["d"] = lambda x, y: (x + y).mean()

        if "h" in self.opts.tasks:
            self.losses["h"] = lambda x, y: (x + y).mean()

        if "s" in self.opts.tasks:
            self.losses["s"] = lambda x, y: (x + y).mean()

        if "w" in self.opts.tasks:
            self.losses["w"] = lambda x, y: (x + y).mean()

        # auto-encoder losses
        self.losses["auto"] = {}
        self.losses["auto"]["a"] = lambda x, y: (x + y).mean()
        self.losses["auto"]["t"] = lambda x, y: (x + y).mean()

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
            domain_batch = {batch["domain"][0]: batch for batch in multi_batch_tuple}
            self.update_g(domain_batch)
            self.update_d(domain_batch)
            self.update_c(domain_batch)

    def train(self):
        assert self.is_setup

        for self.logger.epoch in range(self.opts.train.epochs):
            self.run_epoch()
            self.eval()
            self.save()

    def update_g(self, domain_batch):
        """Perform an update on g from domain_batch which is a dictionary
        domain => batch

        Args:
            domain_batch (dict): dictionnary of domain batches
        """
        r_loss = t_loss = None
        if (
            not self.opts.train.representational_training
            or self.logger.global_step < self.opts.train.representation_steps
        ):
            r_loss = self.get_representation_loss(domain_batch)
        if (
            not self.opts.train.representational_training
            or self.logger.global_step >= self.opts.train.representation_steps
        ):
            t_loss = self.get_translation_loss(domain_batch)

        assert any(l is not None for l in [r_loss, t_loss])

        g_loss = sum(filter(lambda x: x is not None, [r_loss, t_loss]))
        if r_loss is not None:
            self.logger.losses.representation = r_loss.item()
        if t_loss is not None:
            self.logger.losses.translation = t_loss.item()
        self.logger.losses.generator = g_loss.item()
        g_loss.backward()
        self.g_opt_step()

    def get_representation_loss(self, domain_batch):
        """Only update the representation part of the model, meaning everything
        but the translation part

        Args:
            domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders
        """
        step_loss = 0
        self.g_opt.zero_grad()
        for batch_domain, batch in domain_batch.items():
            self.z = self.G.encoder(batch["data"]["x"])
            predictions = {}
            # ---------------------------------------------
            # -----  task-specific regression losses  -----
            # ---------------------------------------------
            for update_task, update_target in batch["data"].items():
                # task t (=translation) will be done in get_translation_loss
                # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                if update_task not in {"t", "a", "x"}:
                    # ? output features classifier
                    predictions[update_task] = self.G.decoders[update_task](self.z)
                    update_loss = self.losses[update_task](
                        update_target, predictions[update_task]
                    )
                    self.logger.task_loss[update_task] = update_loss.item()
                    step_loss += self.opts.train.lambdas[update_task] * update_loss

                    self.debug("get_representation_loss", locals(), 0)

            # --------------------------------------------------
            # -----  auto-encoding update for translation  -----
            # --------------------------------------------------
            translation_decoder = batch_domain[-1]
            reconstruction = self.G.decoders["t"][translation_decoder](self.z)
            update_loss = self.losses["auto"]["t"](batch["data"]["x"], reconstruction)
            self.logger.losses.auto.t = update_loss.item()
            step_loss += self.opts.train.lambdas["auto"]["t"] * update_loss
            self.debug("get_representation_loss", locals(), 1)

            # -------------------------------------------------
            # -----  auto-encoding update for adaptation  -----
            # -------------------------------------------------
            adaptation_decoder = batch_domain[0]
            reconstruction = self.G.decoders["a"][adaptation_decoder](self.z)
            update_loss = self.losses["auto"]["a"](batch["data"]["x"], reconstruction)
            self.logger.losses.auto.a = update_loss.item()
            step_loss += self.opts.train.lambdas.auto.a * update_loss
            self.debug("get_representation_loss", locals(), 2)

        # ----------------------------------------
        # -----  Adversarial adaptaion task  -----
        # ----------------------------------------
        # ? Is this really part of the representation phase
        # TODO include semantic matching loss
        # ? freeze second pass
        # ? noisy labels Alex Lamb ICT
        if "a" in self.opts.tasks:
            adaptation_tasks = []
            if "rn" in domain_batch and "sn" in domain_batch:
                adaptation_tasks.append(("rn", "sn"))
                adaptation_tasks.append(("sn", "rn"))
            if "rf" in domain_batch and "sf" in domain_batch:
                adaptation_tasks.append(("rf", "sf"))
                adaptation_tasks.append(("sf", "rf"))
            # adaptation_tasks = [("rn", "sn"), ("rf", "sf")]
            assert len(adaptation_tasks) > 0

            for source_domain, target_domain in adaptation_tasks:

                real_source = domain_batch[source_domain]["data"]["x"]
                real_target = domain_batch[target_domain]["data"]["x"]

                z_real_source = self.G.encoder(real_source)
                z_real_target = self.G.encoder(real_target)

                fake_source = self.G.decoders["a"][source_domain[0]](z_real_target)
                fake_target = self.G.decoders["a"][target_domain[0]](z_real_source)

                self.debug("get_representation_loss", locals(), 3)

                for real, fake in [  # TODO probably no need for real data  in update
                    (real_source, fake_source),
                    (real_target, fake_target),
                ]:
                    update_loss = self.losses["a"](real, fake)
                    self.logger.losses.a[
                        "{} > {}".format(source_domain, target_domain)
                    ] = update_loss.item()
                    step_loss += self.opts.train.lambdas.a * update_loss
        return step_loss

    def get_translation_loss(self, domain_batch):
        step_loss = 0
        self.g_opt.zero_grad()
        if self.opts.train.freeze_representation:
            freeze(self.G.encoder)
            # ? do I need to also freeze the decoders other than t?

        translation_tasks = []
        if "rn" in domain_batch and "rf" in domain_batch:
            translation_tasks.append(("rn", "rf"))
            translation_tasks.append(("rf", "rn"))
        if "sn" in domain_batch and "sf" in domain_batch:
            translation_tasks.append(("sn", "sf"))
            translation_tasks.append(("sf", "sn"))

        # ? same loop as in representation task but could be different
        # ? when there's the spade components
        for source_domain, target_domain in translation_tasks:
            real_source = domain_batch[source_domain]["data"]["x"]
            real_target = domain_batch[target_domain]["data"]["x"]

            z_real_source = self.G.encoder(real_source)
            z_real_target = self.G.encoder(real_target)

            fake_source = self.G.decoders["t"][source_domain[1]](z_real_target)
            fake_target = self.G.decoders["t"][target_domain[1]](z_real_source)

            self.debug("get_translation_loss", locals())

            for real, fake in [
                (real_source, fake_source),
                (real_target, fake_target),
            ]:
                update_loss = self.losses["t"](real, fake)
                self.logger.losses.t[
                    "{} > {}".format(source_domain, target_domain)
                ] = update_loss.item()
                step_loss += self.opts.train.lambdas.t * update_loss
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
