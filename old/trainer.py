from comet_ml import Experiment
from addict import Dict
from time import time

from .utils import (
    GANLoss,
    get_increasable_name,
    get_optimizer,
    get_scheduler,
    load_conf,
    parsed_args,
    run_val,
    set_requires_grad,
    env_to_path,
    update_learning_rate,
    flatten_conf,
)
from .data import get_all_loaders
from .networks import get_dis, get_flip_res_gen


class Trainer:
    def __init__(self, conf, comet=None, verbose=0):
        super().__init__()

        self.conf = conf
        self.logger = Dict()
        self.logger.lr.g = conf.gen.opt.lr
        self.logger.lr.d = conf.dis.opt.lr
        self.A_loader = self.B_loader = self.val_loaders = None
        self.G = self.D = None

        self.logger.exp = None
        if comet:
            self.logger.exp = Experiment()
            self.logger.exp.log_parameters(flatten_conf(conf))

    def setup(self):
        self.logger.step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.A_loader, self.B_loader, *self.val_loaders = get_all_loaders(
            self.conf.data
        )

        self.G = get_flip_res_gen(self.conf.gen).cuda()
        self.D = get_dis(self.conf.dis).cuda()

        self.g_opt = get_optimizer(self.G, self.conf.gen.opt)
        self.d_opt = get_optimizer(self.D, self.conf.dis.opt)

        if self.verbose > 0:
            print(
                "Loaders: A: {}\nB_loarder: {}\nA_val: {}\nB_val: {}".format(
                    len(self.A_loader),
                    len(self.B_loader),
                    len(self.val_loaders[0]) if self.val_loaders[0] is not None else 0,
                    len(self.val_loaders[1]) if self.val_loaders[1] is not None else 0,
                )
            )

    def run_epoch(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def resume(self):
        pass
