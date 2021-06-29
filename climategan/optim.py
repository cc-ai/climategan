"""Define ExtraAdam and schedulers
"""
import math

import torch
from torch.optim import Adam, Optimizer, RMSprop, lr_scheduler
from torch_optimizer import NovoGrad, RAdam


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """Get an optimizer's learning rate scheduler based on opts

    Args:
        optimizer (torch.Optimizer): optimizer for which to schedule the learning rate
        hyperparameters (addict.Dict): configuration options
        iterations (int, optional): The index of last epoch. Defaults to -1.
            When last_epoch=-1, sets initial lr as lr.

    Returns:
        [type]: [description]
    """

    policy = hyperparameters.get("lr_policy")
    lr_step_size = hyperparameters.get("lr_step_size")
    lr_gamma = hyperparameters.get("lr_gamma")
    milestones = hyperparameters.get("lr_milestones")

    if policy is None or policy == "constant":
        scheduler = None  # constant scheduler
    elif policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma, last_epoch=iterations,
        )
    elif policy == "multi_step":
        if isinstance(milestones, (list, tuple)):
            milestones = milestones
        elif isinstance(milestones, int):
            assert "lr_step_size" in hyperparameters
            if iterations == -1:
                last_milestone = 1000
            else:
                last_milestone = iterations
            milestones = list(range(milestones, last_milestone, lr_step_size))
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_gamma, last_epoch=iterations,
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", hyperparameters["lr_policy"]
        )
    return scheduler


def get_optimizer(net, opt_conf, tasks=None, is_disc=False, iterations=-1):
    """Returns a tuple (optimizer, scheduler) according to opt_conf which
    should come from the trainer's opts as: trainer.opts.<model>.opt

    Args:
        net (nn.Module): Network to update
        opt_conf (addict.Dict): optimizer and scheduler options
        tasks: list of tasks
        iterations (int, optional): Last epoch number. Defaults to -1, meaning
            start with base lr.

    Returns:
        Tuple: (torch.Optimizer, torch._LRScheduler)
    """
    opt = scheduler = None
    lr_names = []
    if tasks is None:
        lr_default = opt_conf.lr
        params = net.parameters()
        lr_names.append("full")
    elif isinstance(opt_conf.lr, float):  # Use default for all tasks
        lr_default = opt_conf.lr
        params = net.parameters()
        lr_names.append("full")
    elif len(opt_conf.lr) == 1:  # Use default for all tasks
        lr_default = opt_conf.lr.default
        params = net.parameters()
        lr_names.append("full")
    else:
        lr_default = opt_conf.lr.default
        params = list()
        for task in tasks:
            lr = opt_conf.lr.get(task, lr_default)
            parameters = None
            # Parameters for encoder
            if not is_disc:
                if task == "m":
                    parameters = net.encoder.parameters()
                    params.append({"params": parameters, "lr": lr})
                    lr_names.append("encoder")
                # Parameters for decoders
                if task == "p":
                    if hasattr(net, "painter"):
                        parameters = net.painter.parameters()
                        lr_names.append("painter")
                else:
                    parameters = net.decoders[task].parameters()
                    lr_names.append(f"decoder_{task}")
            else:
                if task in net:
                    parameters = net[task].parameters()
                    lr_names.append(f"disc_{task}")

            if parameters is not None:
                params.append({"params": parameters, "lr": lr})

    if opt_conf.optimizer.lower() == "extraadam":
        opt = ExtraAdam(params, lr=lr_default, betas=(opt_conf.beta1, 0.999))
    elif opt_conf.optimizer.lower() == "novograd":
        opt = NovoGrad(
            params, lr=lr_default, betas=(opt_conf.beta1, 0)
        )  # default for beta2 is 0
    elif opt_conf.optimizer.lower() == "radam":
        opt = RAdam(params, lr=lr_default, betas=(opt_conf.beta1, 0.999))
    elif opt_conf.optimizer.lower() == "rmsprop":
        opt = RMSprop(params, lr=lr_default)
    else:
        opt = Adam(params, lr=lr_default, betas=(opt_conf.beta1, 0.999))
    scheduler = get_scheduler(opt, opt_conf, iterations)
    return opt, scheduler, lr_names


"""
Extragradient Optimizer

Mostly copied from the extragrad paper repo.

MIT License
Copyright (c) Facebook, Inc. and its affiliates.
written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.
"""


class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.
        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        super(Extragradient, self).__init__(params, defaults)
        self.params_copy = []

    def update(self, p, group):
        raise NotImplementedError

    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current
        parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group["params"]:
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step.
                    # Several extrapolation step can be made before each update but
                    # only the parametersbefore the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError("Need to call extrapolation before calling step.")

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group["params"]:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].add_(u)

        # Free the old parameters
        self.params_copy = []
        return loss


class ExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(ExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError(
                "Adam does not support sparse gradients,"
                + " please consider SparseAdam instead"
            )
        amsgrad = group["amsgrad"]

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = 0
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
        beta1, beta2 = group["betas"]

        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(group["weight_decay"], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)  # type: ignore
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group["eps"])  # type: ignore
        else:
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size * exp_avg / denom
