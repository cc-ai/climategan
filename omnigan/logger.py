import numpy as np
from time import time


def upload_losses(logger):
    if logger.exp is not None:
        logger.exp.log_metrics(logger.losses, step=logger.step)


def print_step(logger):
    s = "e {} | s {} | t/e {:.2f} | t/s {:.2f} | t {:.1f}"
    s += " |#| lr_g {:.2e} lr_d {:.2e}"
    s += " |#| L_G {:.4f} | L_G_rec {:.4f}| L_G_tot {:.4f} | L_D {:.4f}"
    s = s.format(
        logger.epoch + 1,
        logger.step + 1,
        logger.time.epoch,
        logger.time.step,
        logger.time.all,
        logger.lr.g,
        logger.lr.d,
        logger.losses.L_G,
        logger.losses.L_G_rec,
        logger.losses.L_G_tot,
        logger.losses.L_D,
    )
    print("\r" + s, end="")


def set_times(logger):
    logger.time.all = time() - logger.time.start_time
    logger.time.epoch = time() - logger.time.epoch_time

    if logger.time.steps and isinstance(logger.time.steps, list):
        logger.time.steps.append(time() - logger.time.step_time)
        logger.time.steps = logger.time.steps[-100:]
    else:
        logger.time.steps = [time() - logger.time.step_time]

    logger.time.step = np.mean(logger.time.steps)
