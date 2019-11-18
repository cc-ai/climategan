from comet_ml import Experiment
from copy import copy
from pathlib import Path
import torch
from omnigan.data import get_all_loaders
from omnigan.utils import (
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
from omnigan.logger import upload_losses, print_step, set_times
from omnigan.networks import get_dis, get_res_gen, get_flip_res_gen
from addict import Dict
from time import time


def recon_criterion(input, target):
    return torch.mean(torch.abs(input - target))


if __name__ == "__main__":

    # ---------------------------
    # ------ Prepare Params -----
    # ---------------------------

    args = parsed_args()
    logger = Dict()
    logger.step = 0
    start_time = time()
    logger.time.start_time = start_time

    conf = load_conf(Path(__file__).parent.parent / "shared/defaults.yml")
    logger.lr.g = conf.gen.opt.lr
    logger.lr.d = conf.dis.opt.lr

    conf.output_path = env_to_path(conf.output_path)
    conf.output_path = get_increasable_name(conf.output_path)
    Path(conf.output_path).mkdir()
    print("Running model in", conf.output_path)

    logger.exp = None
    if args.comet:
        logger.exp = Experiment()
        logger.exp.log_parameters(flatten_conf(conf))

    # ----------------------------
    # ----- Initialize Model -----
    # ----------------------------

    A_loader, B_loader, *val_loaders = get_all_loaders(conf.data)
    print(
        "A_loader: {}\nB_loarder: {}\nA_val_loader: {}\nB_val_loader: {}".format(
            len(A_loader),
            len(B_loader),
            len(val_loaders[0]) if val_loaders[0] is not None else 0,
            len(val_loaders[1]) if val_loaders[1] is not None else 0,
        )
    )

    # G = get_res_gen(conf.gen).cuda()
    G = get_flip_res_gen(conf.gen).cuda()
    D = get_dis(conf.dis).cuda()

    g_opt = get_optimizer(G, conf.gen.opt)
    d_opt = get_optimizer(D, conf.dis.opt)

    g_scheduler = get_scheduler(g_opt, conf.gen.opt)
    d_scheduler = get_scheduler(d_opt, conf.gen.opt)

    ganloss = GANLoss("lsgan").cuda()

    # ----------------------
    # ----- Train Loop -----
    # ----------------------

    for epoch in range(conf.train.epochs):
        logger.epoch = epoch
        logger.time.epoch_time = time()
        for i, (real_im_A, real_im_B) in enumerate(zip(B_loader, A_loader)):

            # ------------------
            # ----- Set up -----
            # ------------------

            logger.time.step_time = time()
            logger.epoch_step = i
            if conf.data.return_paths:
                real_im_A, real_im_A_path_list = real_im_A
                real_im_B, real_im_B_path_list = real_im_B

            real_im_A = real_im_A.cuda()
            real_im_B = real_im_B.cuda()

            # ---------------------
            # ----- Forward G -----
            # ---------------------

            fake_im_AB = G(real_im_A, flip_val=-1)
            fake_im_BA = G(real_im_B, flip_val=1)

            fake_im_AA = G(real_im_A, flip_val=1)
            fake_im_BB = G(real_im_B, flip_val=-1)

            fake_im_ABA = G(fake_im_AB, flip_val=1)
            fake_im_BAB = G(fake_im_BA, flip_val=-1)

            # ----------------------
            # ----- Backward D -----
            # ----------------------

            set_requires_grad([D], True)
            d_opt.zero_grad()

            real_prob_A = D(real_im_A)
            fake_prob_A = D(fake_im_BA.detach())
            fake_prob_BA = D(fake_im_ABA.detach())

            real_prob_B = D(real_im_B)
            fake_prob_B = D(fake_im_AB.detach())
            fake_prob_AB = D(fake_im_BAB.detach())

            L_D = (
                ganloss(real_prob_A, True)
                + ganloss(real_prob_B, True)
                + ganloss(fake_prob_A, False)
                + ganloss(fake_prob_B, False)
                + ganloss(fake_prob_AB, False)
                + ganloss(fake_prob_BA, False)
            )

            logger.losses.L_D = L_D.item()
            L_D.backward()

            d_opt.step()

            # ----------------------
            # ----- Backward G -----
            # ----------------------

            set_requires_grad([D], False)
            g_opt.zero_grad()

            L_G = (
                ganloss(D(fake_im_BA), True)
                + ganloss(D(fake_im_AB), True)
                + ganloss(D(fake_im_BAB), True)
                + ganloss(D(fake_im_ABA), True)
            )

            L_G_rec = recon_criterion(real_im_A, fake_im_ABA) + recon_criterion(
                real_im_B, fake_im_BAB
            )  # Reconstruction loss going to a domain and back
            L_G_idt = recon_criterion(real_im_A, fake_im_AA) + recon_criterion(
                real_im_B, fake_im_BB
            )  # Identity loss going to a domain where the image already is

            L_G_tot = (
                L_G + conf.gen.lambda_rec * L_G_rec + conf.gen.lambda_idt * L_G_idt
            )

            logger.losses.L_G = L_G.item()
            logger.losses.L_G_rec = L_G_rec.item()
            logger.losses.L_G_idt = L_G_idt.item()
            logger.losses.L_G_tot = L_G_tot.item()

            L_G_tot.backward()
            g_opt.step()

            # -------------------
            # ----- Logging -----
            # -------------------

            upload_losses(logger)
            set_times(logger)
            print_step(logger)
            run_val(G, val_loaders, conf, logger)
            logger.step += 1
        # ------------------------
        # ----- End of Epoch -----
        # ------------------------
        print()
        update_learning_rate(g_scheduler, g_opt, conf.gen.opt, logger, True)
        update_learning_rate(d_scheduler, d_opt, conf.dis.opt, logger, False)

    # -----------------------------
    # ----- End of Train Loop -----
    # -----------------------------
