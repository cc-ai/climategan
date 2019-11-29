import os
import re
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from copy import copy
import numpy as np
import torch
import yaml
from addict import Dict
from PIL import Image
from torch.optim import lr_scheduler


def load_opts(path):
    path = Path(path).resolve()
    print("Loading opts from", str(path))
    with open(path, "r") as stream:
        try:
            opts = Dict(yaml.safe_load(stream))
            for k in opts.gen.decoders:
                tmp = copy(opts.gen.default)
                if k in opts.gen:
                    tmp.update(opts.gen[k])
                opts.gen[k] = tmp
            return opts
        except yaml.YAMLError as exc:
            print(exc)


def get_optimizer(net, conf_net_opt):
    return torch.optim.Adam(
        net.parameters(), lr=conf_net_opt.lr, betas=(conf_net_opt.beta1, 0.999)
    )


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def update_learning_rate(scheduler, optimizer, conf_net_opt, logger, is_G):
    """Update learning rates for all the networks; called at the end of every epoch"""
    if conf_net_opt.lr_policy is False:
        return
    if conf_net_opt.lr_policy == "plateau":
        scheduler.step(0)
    else:
        scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    if is_G:
        logger.lr.g = lr
    else:
        logger.lr.d = lr


def get_scheduler(optimizer, conf_net_opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass
                              of BaseOptions．　
                              conf_net_opt.lr_policy is the name of learning rate
                                        lsgan, and wgangp.
                              policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine),
    we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if conf_net_opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + conf_net_opt.epoch_count - conf_net_opt.niter
            ) / float(conf_net_opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif conf_net_opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=conf_net_opt.lr_decay_iters, gamma=0.1
        )
    elif conf_net_opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif conf_net_opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=conf_net_opt.niter, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", conf_net_opt.lr_policy
        )
    return scheduler


def parsed_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="What configuration file to use to overwrite shared/defaults.yml",
    )
    parser.add_argument(
        "--comet", action="store_true", help="Use comet.ml to log experiment"
    )
    return parser.parse_args()


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def run_val(G, val_loaders, opts, logger):
    if not any(val_loaders):
        return
    if not opts.val:
        return
    if not logger.step % opts.val.every_n_steps == 0:
        return

    with Path(opts.output_path / "hash.txt").open("w") as f:
        f.write(get_git_revision_hash())

    G.eval()

    print("\nRunning evaluation...", end="")
    A_val_loader, B_val_loader = val_loaders
    if A_val_loader is None:
        A_val_loader = [None] * len(B_val_loader)
    if B_val_loader is None:
        B_val_loader = [None] * len(A_val_loader)

    num_images = 0

    for i, (real_im_A, real_im_B) in enumerate(zip(A_val_loader, B_val_loader)):

        num_images += opts.data.loaders.batch_size

        if real_im_A is not None:
            if opts.data.return_paths:
                real_im_A, real_im_A_path_list = real_im_A
            real_im_A = real_im_A.cuda()
            fake_im_AB = G(real_im_A, -1)
            imgs_A = [real_im_A, fake_im_AB]
            if opts.val.infer_rec:
                imgs_A.append(G(fake_im_AB, 1))
            if opts.val.infer_idt:
                imgs_A.append(G(real_im_A, 1))
            save_imgs(imgs_A, logger, opts, True, True, i)

        if real_im_B is not None:
            if opts.data.return_paths:
                real_im_B, real_im_B_path_list = real_im_B

            real_im_B = real_im_B.cuda()
            fake_im_BA = G(real_im_B, 1)
            imgs_B = [real_im_B, fake_im_BA]
            if opts.val.infer_rec:
                imgs_B.append(G(fake_im_BA, -1))
            if opts.val.infer_idt:
                imgs_B.append(G(real_im_B, -1))
            save_imgs(imgs_B, logger, opts, True, True, i)

        if num_images >= opts.val.max_log_images:
            break
    print(
        "Done, ran on {} images{}.".format(
            num_images, "" if logger.exp is None else " and uploaded them"
        )
    )
    G.train()


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor.cpu().detach().float().numpy()
        )  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_imgs(imgs_tensors, logger, opts, is_A, is_val, iter):
    """Saves images to disk in opts.output_path / images / {train or val}
    then uploads horizontally-stacked images to comet if logger.exp is not None

    Args:
        real_tensor (torch.Tensor): contains the batch data for the loader's
        real image
        fake_tensor (torch.Tensor): contains the batch images output by the Generator
        logger (addict.Dict): Logger
        opts (addict.Dict): Configuration dict
        is_A (bool): Select domain, A or B (for image names)
        is_val (bool): Select mode, train or val (for image names)
    """
    npys = np.array([[tensor2im(i) for i in tensor] for tensor in imgs_tensors])
    npys = [npys[:, i, :, :, :] for i in range(npys.shape[1])]

    for i, npy in enumerate(npys):
        idx = iter * len(imgs_tensors[0]) + i
        stacked_npy = np.concatenate(npy, axis=1)

        # real_image = Image.fromarray(real_npy)
        # fake_image = Image.fromarray(fake_npy)
        staked_image = Image.fromarray(stacked_npy)

        im_path = Path(opts.output_path) / "images" / "val" / str(logger.epoch)
        im_path.mkdir(exist_ok=True, parents=True)
        # real_image_path = im_path / "{}_{}_{}_{}_{}.png".format(
        #     "A" if is_A else "B",
        #     "val" if is_val else "train",
        #     "real", idx, logger.step
        # )
        # fake_image_path = im_path / "{}_{}_{}_{}_{}.png".format(
        #     "A" if is_A else "B",
        #     "val" if is_val else "train",
        #     "fake", idx, logger.step
        # )
        staked_image_path = im_path / "stacked_{}_{}_{}_{}.png".format(
            "A" if is_A else "B", "val" if is_val else "train", idx, logger.step
        )

        if opts.val.store_images:
            # real_image.save(str(real_image_path))
            # fake_image.save(str(fake_image_path))
            staked_image.save(str(staked_image_path))

        if logger.exp is not None:
            logger.exp.log_image(
                staked_image,
                "{}_{}_{}.png".format(
                    "A" if is_A else "B", "val" if is_val else "train", logger.step
                ),
                step=logger.step,
            )


def get_increasable_name(file_path):
    f = Path(file_path)
    while f.exists():
        name = f.name
        s = list(re.finditer(r"(\d+)", name))
        if s:
            s = s[-1]
            d = int(s.group().replace("(", "").replace(")", "").replace(".", ""))
            d += 1
            i, j = s.span()
            name = name[:i] + f"{d}" + name[j:]
        else:
            name = f.stem + " (1)" + f.suffix
        f = f.parent / name
    return f


def env_to_path(path):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable

    """
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


def flatten_opts(opts):
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, Dict):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                for i, m in enumerate(v):
                    p(m, prefix + k + "." + str(i) + ".", vals)
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)
