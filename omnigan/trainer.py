from comet_ml import Experiment
from addict import Dict
from time import time

from .generator import get_gen
from .utils import flatten_conf
from .optim import get_optimizer
from .data import get_all_loaders


class Trainer:
    def __init__(self, conf, comet=None, verbose=0):
        super().__init__()

        self.conf = conf
        self.logger = Dict()
        self.logger.lr.g = conf.gen.opt.lr
        self.logger.lr.d = conf.dis.opt.lr
        self.loaders = None
        self.G = self.D = None

        self.is_setup = False

        self.logger.exp = None
        if comet:
            self.logger.exp = Experiment()
            self.logger.exp.log_parameters(flatten_conf(conf))

    def setup(self):
        self.logger.step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.loaders = get_all_loaders(self.conf.data)

        self.G = get_gen(self.conf.gen).cuda()
        self.D = get_dis(self.conf.dis).cuda()
        self.C = get_classifier(self.conf.classifier)  # TODO

        self.g_opt = get_optimizer(self.G, self.conf.gen.opt)
        self.d_opt = get_optimizer(self.D, self.conf.dis.opt)

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

        for self.logger.epoch in range(self.conf.train.epochs):
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
