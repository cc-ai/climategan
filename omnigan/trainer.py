from comet_ml import Experiment
from addict import Dict
from time import time
import torch

from omnigan.generator import get_gen
from omnigan.utils import flatten_opts
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
            self.losses["a"] = lambda x: -1

        if "t" in self.opts.tasks:
            self.losses["a"] = lambda x: -1

        # task losses
        if "d" in self.opts.tasks:
            self.losses["d"] = lambda x: -1

        if "h" in self.opts.tasks:
            self.losses["a"] = lambda x: -1

        if "s" in self.opts.tasks:
            self.losses["a"] = lambda x: -1

        if "w" in self.opts.tasks:
            self.losses["a"] = lambda x: -1

        # auto-encoder losses
        self.losses["auto"]["a"] = lambda x: -1
        self.losses["auto"]["t"] = lambda x: -1

    def setup(self):
        self.logger.step = 0
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

    def run_epoch(self):
        assert self.is_setup
        for i, multi_batch_tuple in enumerate(
            zip(*[self.loaders["train"][domain] for domain in self.loaders["train"]])
        ):
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
        if self.logger.steps > self.opts.train.represention_steps:
            self.update_g_representation(domain_batch)
        else:
            self.update_g_translation(domain_batch)

    def update_g_representation(self, domain_batch):
        step_loss = 0
        for batch in domain_batch.values():
            self.z = self.G.encoder(batch["data"]["x"])
            self.predictions = {}
            domain = batch["domain"][0]
            # task-specific regression losses
            for task, target in batch["data"].items():
                # task t (=translation) will be done in update_g_translation
                # task a (= adaptation) will be done hereafter
                if task in {"t", "a", "x"}:
                    continue
                else:
                    self.predictions[task] = self.G.decoders[task](self.z)
                    loss = self.losses[task](target, self.predictions[task])
                    step_loss += self.opts.train.lambdas[task] * loss

            # auto-encoding update for translation
            translation_decoder = domain[-1]
            reconstruction = self.G.decoders["t"][translation_decoder](self.z)
            loss = self.losses["auto"]["t"](target, reconstruction)
            step_loss += self.opts.train.lambdas["auto"]["t"] * loss

            # auto-encoding update for adaptation
            adaptation_decoder = domain[0]
            reconstruction = self.G.decoders["a"][adaptation_decoder](self.z)
            loss = self.losses["auto"]["a"](target, reconstruction)
            step_loss += self.opts.train.lambdas["auto"]["a"] * loss

        if "a" in self.opts.tasks:
            # adaptaion task
            if "rn" in domain_batch and "sn" in domain_batch:
                self.z_rn = self.G.encoder(domain_batch["rn"]["data"]["x"])
                self.fake_sn = self.G.A["s"](self.z_rn)

                self.z_sn = self.G.encoder(domain_batch["rn"]["data"]["x"])
                self.fake_sn = self.G.A["s"](self.z_rn)

            if "rf" in domain_batch and "sf" in domain_batch:
                pass

    def update_g_translation(self, batch):
        pass

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
