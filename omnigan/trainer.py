from comet_ml import Experiment
from addict import Dict
from time import time

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
        self.logger = Dict()
        self.logger.lr.g = opts.gen.opt.lr
        self.logger.lr.d = opts.dis.opt.lr
        self.loaders = None
        self.G = self.D = None

        self.is_setup = False

        self.logger.exp = None
        if comet:
            self.logger.exp = Experiment()
            self.logger.exp.log_parameters(flatten_opts(opts))

    def compute_latent_shape(self):
        pass

    def setup(self):
        self.logger.step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.loaders = get_all_loaders(self.opts.data)

        self.G = get_gen(self.opts.gen).cuda()
        self.D = get_dis(self.opts.dis).cuda()
        self.latent_shape = self.compute_latent_shape()  # TODO
        self.C = get_classifier(self.opts.classifier, self.latent_shape)

        self.g_opt = get_optimizer(self.G, self.opts.gen.opt)
        self.d_opt = get_optimizer(self.D, self.opts.dis.opt)

        if self.verbose > 0:
            for mode, mode_dict in self.loaders.items():
                for domain, domain_loader in mode_dict:
                    print("Loader {} {} : {}".format(mode, domain, len(domain_loader)))

        self.is_setup = True

    def run_epoch(self):
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
        loss = 0

        if "H" in batch["tasks"]:
            for image in batch["images"].items():
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
