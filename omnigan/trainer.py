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
        self.G = self.D = None

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

        print(b)

        b = self.batch_to_device(b)
        z = self.G.encoder(b.data.x)
        return z.shape[1:]

    def setup(self):
        self.logger.step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.loaders = get_all_loaders(self.opts)

        self.G = get_gen(self.opts, verbose=self.verbose).to(self.device)
        self.D = get_dis(self.opts, verbose=self.verbose).to(self.device)
        self.latent_shape = self.compute_latent_shape()  # TODO
        self.C = get_classifier(self.opts, self.latent_shape, verbose=self.verbose).to(
            self.device
        )

        self.g_opt = get_optimizer(self.G, self.opts.gen.opt)
        self.d_opt = get_optimizer(self.D.models, self.opts.dis.opt)

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
        for i, self.batch in enumerate(self.train_loaders):
            self.logger.step += 1

            self.update_g(self.batch)
            self.update_d(self.batch)
            self.update_c(self.batch)

    def train(self):
        assert self.is_setup

        for self.logger.epoch in range(self.opts.train.epochs):
            self.run_epoch()
            self.eval()
            self.save()

    def update_g(self, batch):
        if self.logger.steps > self.opts.train.represention_steps:
            self.update_g_representation(batch)
        else:
            self.update_g_translation(batch)

    def update_g_representation(self, batch):
        self.z = self.G.encoder(batch.data.x)

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
