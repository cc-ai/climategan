from comet_ml import Experiment
from addict import Dict
from time import time

from .utils import (
    GANLoss,
    get_increasable_name,
    get_optimizer,
    get_scheduler,
    load_opts,
    parsed_args,
    run_val,
    set_requires_grad,
    env_to_path,
    update_learning_rate,
    flatten_opts,
)
from .data import get_all_loaders
from .networks import get_dis, get_flip_res_gen


class Trainer:
    def __init__(self, opts, comet=None, verbose=0):
        super().__init__()

        self.opts = opts
        self.logger = Dict()
        self.logger.lr.g = opts.gen.opt.lr
        self.logger.lr.d = opts.dis.opt.lr
        self.A_loader = self.B_loader = self.val_loaders = None
        self.G = self.D = None

        self.logger.exp = None
        if comet:
            self.logger.exp = Experiment()
            self.logger.exp.log_parameters(flatten_opts(opts))

    def setup(self):
        self.logger.step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.A_loader, self.B_loader, *self.val_loaders = get_all_loaders(
            self.opts.data
        )

        self.G = get_flip_res_gen(self.opts.gen).cuda()
        self.D = get_dis(self.opts.dis).cuda()

        self.g_opt = get_optimizer(self.G, self.opts.gen.opt)
        self.d_opt = get_optimizer(self.D, self.opts.dis.opt)

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
