"""Main component: the trainer handles everything:
    * initializations
    * training
    * saving
"""
from comet_ml import Experiment
from time import time
import torch
from addict import Dict
from pathlib import Path

from omnigan.classifier import get_classifier
from omnigan.data import get_all_loaders
from omnigan.discriminator import get_dis
from omnigan.generator import get_gen
from omnigan.losses import get_losses
from omnigan.optim import get_optimizer
from omnigan.mega_depth import get_mega_model
from omnigan.utils import flatten_opts
from omnigan.tutils import (
    domains_to_class_tensor,
    fake_domains_to_class_tensor,
    freeze,
    shuffle_batch_tuple,
    get_num_params,
    vgg_preprocess,
)
import torchvision.utils as vutils
import os


class Trainer:
    """Main trainer class
    """

    def __init__(self, opts, comet_exp=None, verbose=0):
        """Trainer class to gather various model training procedures
        such as training evaluating saving and logging

        init:
        * creates an addict.Dict logger
        * creates logger.exp as a comet_exp experiment if `comet` arg is True
        * sets the device (1 GPU or CPU)

        Args:
            opts (addict.Dict): options to configure the trainer, the data, the models
            comet (bool, optional): whether to log the trainer with comet.ml.
                                    Defaults to False.
            verbose (int, optional): printing level to debug. Defaults to 0.
        """
        super().__init__()

        self.opts = opts
        self.verbose = verbose
        self.logger = Dict()
        self.logger.lr.g = opts.gen.opt.lr
        self.logger.lr.d = opts.dis.opt.lr
        self.logger.epoch = 0
        self.loaders = None
        self.losses = None

        self.is_setup = False
        self.representation_is_frozen = False
        self.translation_map = {"r": "s", "s": "r", "f": "n", "n": "f"}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.exp = None
        if isinstance(comet_exp, Experiment):
            self.exp = comet_exp
        self.source_label = 0
        self.target_label = 1

    def log_losses(self, model_to_update="G", mode="train"):
        """Logs metrics on comet.ml

        Args:
            model_to_update (str, optional): One of "G", "D" or "C". Defaults to "G".
        """
        loss_names = {"G": "generator", "D": "discriminator", "C": "classifier"}

        if self.opts.train.log_level < 1:
            return

        if self.exp is None:
            return

        assert model_to_update in {
            "G",
            "D",
            "C",
        }, "unknown model to log losses {}".format(model_to_update)

        loss_to_update = self.logger.losses[loss_names[model_to_update]]

        losses = loss_to_update.copy()

        if self.opts.train.log_level == 1:
            # Only log aggregated losses: delete other keys in losses
            for k in self.logger.losses:
                if k not in {"representation", "generator", "painter"}:
                    del losses[k]
        # convert losses into a single-level dictionnary

        losses = flatten_opts(losses)
        self.exp.log_metrics(
            losses, prefix=f"{model_to_update}_{mode}", step=self.logger.global_step
        )

    def batch_to_device(self, b):
        """sends the data in b to self.device

        Args:
            b (dict): the batch dictionnay

        Returns:
            dict: the batch dictionnary with its "data" field sent to self.device
        """
        for task, tensor in b["data"].items():
            b["data"][task] = tensor.to(self.device)
        return b

    def compute_latent_shape(self):
        """Compute the latent shape, i.e. the Encoder's output shape,
        from a batch.

        Raises:
            ValueError: If no loader, the latent_shape cannot be inferred

        Returns:
            tuple: (c, h, w)
        """
        b = None
        for mode in self.loaders:
            for domain in self.loaders[mode]:
                b = Dict(next(iter(self.loaders[mode][domain])))
                break
        if b is None:
            raise ValueError("No batch found to compute_latent_shape")
        b = self.batch_to_device(b)
        z = self.G.encode(b.data.x)
        return z.shape[1:]

    def compute_input_shape(self):
        """Compute the latent shape, i.e. the Encoder's output shape,
        from a batch.

        Raises:
            ValueError: If no loader, the latent_shape cannot be inferred

        Returns:
            tuple: (c, h, w)
        """
        b = None
        for mode in self.loaders:
            for domain in self.loaders[mode]:
                b = Dict(next(iter(self.loaders[mode][domain])))
                break
        if b is None:
            raise ValueError("No batch found to compute_latent_shape")
        b = self.batch_to_device(b)
        return b.data.x.shape[1:]

    def setup(self):
        """Prepare the trainer before it can be used to train the models:
            * initialize G and D
            * compute latent space dims and create classifier accordingly
            * creates 3 optimizers
        """
        self.logger.global_step = 0
        start_time = time()
        self.logger.time.start_time = start_time

        self.loaders = get_all_loaders(self.opts)

        self.G = get_gen(self.opts, verbose=self.verbose).to(self.device)
        self.latent_shape = self.compute_latent_shape()
        self.input_shape = self.compute_input_shape()
        self.painter_z_h = self.input_shape[-2] // (2 ** self.opts.gen.p.spade_n_up)
        self.painter_z_w = self.input_shape[-1] // (2 ** self.opts.gen.p.spade_n_up)

        self.D = get_dis(self.opts, verbose=self.verbose).to(self.device)
        self.C = get_classifier(self.opts, self.latent_shape, verbose=self.verbose).to(
            self.device
        )
        self.P = {"s": get_mega_model()}  # P => pseudo labeling models

        self.g_opt, self.g_scheduler = get_optimizer(self.G, self.opts.gen.opt)

        print("---------------------------")
        print("num params encoder: ", get_num_params(self.G.encoder))
        print("num params decoder: ", get_num_params(self.G.decoders["m"]))
        print("num params painter: ", get_num_params(self.G.painter))
        print("num params classif: ", get_num_params(self.C))
        print("num params discrim: ", get_num_params(self.D))
        print("---------------------------")

        if get_num_params(self.D) > 0:
            self.d_opt, self.d_scheduler = get_optimizer(self.D, self.opts.dis.opt)
        else:
            self.d_opt, self.d_scheduler = None, None
        self.c_opt, self.c_scheduler = get_optimizer(self.C, self.opts.classifier.opt)

        if self.opts.train.resume:
            self.resume()

        self.losses = get_losses(self.opts, self.verbose, device=self.device)

        if self.verbose > 0:
            for mode, mode_dict in self.loaders.items():
                for domain, domain_loader in mode_dict.items():
                    print(
                        "Loader {} {} : {}".format(
                            mode, domain, len(domain_loader.dataset)
                        )
                    )

        # Create display images:
        print("Creating display images...", end="", flush=True)

        if type(self.opts.comet.display_size) == int:
            display_indices = range(self.opts.comet.display_size)
        else:
            display_indices = self.opts.comet.display_size

        self.display_images = {}
        for mode, mode_dict in self.loaders.items():
            self.display_images[mode] = {}
            for domain, domain_loader in mode_dict.items():

                self.display_images[mode][domain] = [
                    Dict(self.loaders[mode][domain].dataset[i])
                    for i in display_indices
                    if i < len(self.loaders[mode][domain].dataset)
                ]

        self.is_setup = True

    def g_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.gen.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.g_opt.extrapolation()
        else:
            self.g_opt.step()

    def d_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.dis.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.d_opt.extrapolation()
        else:
            self.d_opt.step()

    def c_opt_step(self):
        """Run an optimizing step ; if using ExtraAdam, there needs to be an extrapolation
        step every other step
        """
        if "extra" in self.opts.classifier.opt.optimizer.lower() and (
            self.logger.global_step % 2 == 0
        ):
            self.c_opt.extrapolation()
        else:
            self.c_opt.step()

    @property
    def train_loaders(self):
        """Get a zip of all training loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["train"].values()))

    def update_learning_rates(self):
        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()
        if self.c_scheduler is not None:
            self.c_scheduler.step()

    @property
    def val_loaders(self):
        """Get a zip of all validation loaders

        Returns:
            generator: zip generator yielding tuples:
                (batch_rf, batch_rn, batch_sf, batch_sn)
        """
        return zip(*list(self.loaders["val"].values()))

    def run_epoch(self):
        """Runs an epoch:
        * checks trainer is setup
        * gets a tuple of batches per domain
        * sends batches to device
        * updates sequentially G, D, C
        """
        assert self.is_setup

        for i, multi_batch_tuple in enumerate(self.train_loaders):
            # create a dictionnay (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device
            print(
                "\rEpoch {} batch {} step {}".format(
                    self.logger.epoch, i, self.logger.global_step
                )
            )

            step_start_time = time()
            multi_batch_tuple = shuffle_batch_tuple(multi_batch_tuple)
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }

            # freeze params of advent discriminator
            for param in self.D["m"]["maskAdvent"].parameters():
                param.requires_grad = False

            self.update_g(multi_domain_batch)
            if self.d_opt is not None:
                # unfreeze params of advent discriminator
                for param in self.D["m"]["maskAdvent"].parameters():
                    param.requires_grad = True

                self.update_d(multi_domain_batch)
            self.update_c(multi_domain_batch)
            self.logger.global_step += 1
            if self.should_freeze_representation():
                freeze(self.G.encoder)
                # ? Freeze decoders != t for memory management purposes ; faster ?
                self.representation_is_frozen = True
            step_time = time() - step_start_time
            self.log_step_time(step_time)

        if self.opts.art == "mask":
            self.log_comet_images("train", "r")
            self.log_comet_images("train", "s")
        elif self.opts.art == "paint":
            self.log_comet_images("train", "rf")
        else:
            raise ValueError("Unknown opts.art {}".format(self.opts.art))

        self.update_learning_rates()

    def log_step_time(self, step_time):
        """Logs step-time on comet.ml

        Args:
            step_time (float): step-time in seconds
        """
        if self.exp:
            self.exp.log_metric("Step-time", step_time, step=self.logger.global_step)

    def log_comet_images(self, mode, domain):

        save_images = {}
        if domain != "rf":
            for im_set in self.display_images[mode][domain]:
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)

                self.z = self.G.encode(x)

                for update_task, update_target in im_set["data"].items():
                    target = im_set["data"][update_task].unsqueeze(0).to(self.device)
                    task_saves = []
                    if update_task != "x":
                        if update_task not in save_images:
                            save_images[update_task] = []
                        prediction = self.G.decoders[update_task](self.z)

                        if update_task in {"m"}:
                            prediction = prediction.repeat(1, 3, 1, 1)
                            task_saves.append(x * (1.0 - prediction))
                            task_saves.append(x * (1.0 - target.repeat(1, 3, 1, 1)))
                        task_saves.append(prediction)
                        # ! This assumes the output is some kind of image
                        save_images[update_task].append(x)
                        for im in task_saves:
                            save_images[update_task].append(im)

            for task in save_images.keys():
                # Write images:
                self.write_images(
                    image_outputs=save_images[task],
                    mode=mode,
                    domain=domain,
                    task=task,
                    im_per_row=4,
                    comet_exp=self.exp,
                )
        else:
            image_outputs = []
            for im_set in self.display_images[mode][domain]:
                x = im_set["data"]["x"].unsqueeze(0).to(self.device)
                m = im_set["data"]["m"].unsqueeze(0).to(self.device)

                batch_size = x.shape[0]
                z = (
                    torch.empty(
                        batch_size,
                        self.opts.gen.p.latent_dim,
                        self.painter_z_h,
                        self.painter_z_w,
                    )
                    .normal_(mean=0, std=1.0)
                    .to(self.device)
                )
                prediction = self.G.painter(z, x * (1.0 - m))
                image_outputs.append(x * (1.0 - m))
                image_outputs.append(prediction)
                image_outputs.append(x)
                image_outputs.append(prediction * m)
            # Write images
            self.write_images(
                image_outputs=image_outputs,
                mode=mode,
                domain=domain,
                task="painter",
                im_per_row=4,
                comet_exp=self.exp,
            )
        return 0

    def write_images(
        self, image_outputs, mode, domain, task, im_per_row=3, comet_exp=None
    ):
        """Save output image
        Arguments:
            image_outputs {Tensor list} -- list of output images
            im_per_row {int} -- number of images to be displayed (per row)
            file_name {str} -- name of the file where to save the images
        """
        curr_iter = self.logger.global_step
        image_outputs = torch.stack(image_outputs).squeeze()
        image_grid = vutils.make_grid(
            image_outputs, nrow=im_per_row, normalize=True, scale_each=True
        )
        image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()

        if comet_exp is not None:
            comet_exp.log_image(
                image_grid,
                name=f"{mode}_{domain}_{task}_{str(curr_iter)}",
                step=curr_iter,
            )

    def train(self):
        """For each epoch:
        * train
        * eval
        * save
        """
        assert self.is_setup

        for self.logger.epoch in range(
            self.logger.epoch, self.logger.epoch + self.opts.train.epochs
        ):
            self.run_epoch()
            self.eval(verbose=1)
            if (
                self.logger.epoch != 0
                and self.logger.epoch % self.opts.train.save_n_epochs == 0
            ):
                self.save()

    def should_freeze_representation(self):
        if self.representation_is_frozen:
            return False
        if not self.opts.train.freeze_representation:
            return False
        if not self.opts.train.representational_training:
            return False
        if self.logger.global_step < self.opts.train.representation_steps:
            return False
        return True

    def should_compute_r_loss(self):
        if not self.opts.train.representational_training:
            return True
        if self.logger.global_step < self.opts.train.representation_steps:
            return True
        return False

    def should_compute_t_loss(self):
        if not self.opts.train.representational_training:
            return True
        if self.logger.global_step >= self.opts.train.representation_steps:
            return True
        return False

    def update_g(self, multi_domain_batch, verbose=0):
        """Perform an update on g from multi_domain_batch which is a dictionary
        domain => batch

        * compute loss
            * if using Sam Lavoie's representational_training:
                * compute either representation_loss or translation_loss
                  depending on the current step vs opts.train.representation_steps
            * otherwise compute both
        * loss.backward()
        * g_opt_step()
            * g_opt.step() or .extrapolation() depending on self.logger.global_step
        * logs losses on comet.ml with self.log_losses(model_to_update="G")

        Args:
            multi_domain_batch (dict): dictionnary of domain batches
        """
        self.g_opt.zero_grad()
        r_loss = p_loss = None

        # For now, always compute "representation loss"
        if self.opts.art == "mask":
            r_loss = self.get_representation_loss(multi_domain_batch)

        if self.opts.art == "paint":
            p_loss = self.get_painter_loss(multi_domain_batch)

        # if self.should_compute_t_loss():
        #    t_loss = self.get_translation_loss(multi_domain_batch)

        assert any(l is not None for l in [r_loss, p_loss]), "Both losses are None"

        g_loss = 0
        if r_loss is not None:
            g_loss += r_loss
            if verbose > 0:
                print("adding r_loss {} to g_loss".format(r_loss))
            self.logger.losses.representation = r_loss.item()

        if p_loss is not None:
            g_loss += p_loss
            if verbose > 0:
                print("adding p_loss {} to g_loss".format(p_loss))
            self.logger.losses.painter = p_loss.item()

        if verbose > 0:
            print("g_loss is {}".format(g_loss))

        self.logger.losses.generator.total_loss = g_loss.item()
        g_loss.backward()
        self.g_opt_step()
        self.log_losses(model_to_update="G", mode="train")

    def get_representation_loss(self, multi_domain_batch):
        """Only update the representation part of the model, meaning everything
        but the translation part

        * for each batch in available domains:
            * compute latent classifier loss with fake labels(1)
            * compute task-specific losses (2)
            * compute the adaptation and translation decoders' auto-encoding losses (3)
            * compute the adaptation decoder's translation losses (GAN and Cycle) (4)

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        step_loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        # ? should we add all domains to the loss (.backward() and .step() after this
        # ? loop) or update the networks for each domain sequentially
        # ? (.backward() and .step() n times)?
        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain == "rf":
                continue

            x = batch["data"]["x"]
            self.z = self.G.encode(x)
            # ---------------------------------
            # -----  classifier loss (1)  -----
            # ---------------------------------
            # Forward pass through classifier, output : (batch_size, 4)
            output_classifier = self.C(self.z)

            # Cross entropy loss (with sigmoid) with fake labels to fool C
            update_loss = self.losses["G"]["classifier"](
                output_classifier,
                fake_domains_to_class_tensor(batch["domain"], one_hot),
            )

            step_loss += lambdas.G.classifier * update_loss

            # -------------------------------------------------
            # -----  task-specific regression losses (2)  -----
            # -------------------------------------------------
            task_tensors = {}
            for update_task, update_target in batch["data"].items():
                # task t (=translation) will be done in get_translation_loss
                # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                if update_task not in {"t", "a", "x", "m"}:
                    # ? output features classifier
                    prediction = self.G.decoders[update_task](self.z)
                    task_tensors[update_task] = prediction
                    update_loss = self.losses["G"]["tasks"][update_task](
                        prediction, update_target
                    )

                    step_loss += lambdas.G[update_task] * update_loss
                    self.logger.losses.generator.task_loss[update_task][
                        batch_domain
                    ] = update_loss.item()
                if update_task == "m":
                    # ? output features classifier
                    prediction = self.G.decoders[update_task](self.z)
                    task_tensors[update_task] = prediction

                    # Main loss first:
                    update_loss = (
                        self.losses["G"]["tasks"][update_task]["main"](
                            prediction, update_target
                        )
                        * lambdas.G[update_task]["main"]
                    )
                    step_loss += update_loss

                    self.logger.losses.generator.task_loss[update_task]["main"][
                        batch_domain
                    ] = update_loss.item()

                    # Then TV loss
                    update_loss = self.losses["G"]["tasks"][update_task]["tv"](
                        prediction
                    )
                    step_loss += update_loss

                    self.logger.losses.generator.task_loss[update_task]["tv"][
                        batch_domain
                    ] = update_loss.item()

                    # Then Advent loss
                    if batch_domain == "r":
                        pred_prime = 1 - prediction
                        prob = torch.cat([prediction, pred_prime], dim=1)

                        update_loss = self.losses["G"]["tasks"]["m"]["advent"](
                            prob.to(self.device),
                            self.source_label,
                            self.D["m"]["maskAdvent"],
                        )
                    step_loss += update_loss

                    self.logger.losses.generator.task_loss[update_task]["advent"][
                        batch_domain
                    ] = update_loss.item()
        return step_loss

    def get_painter_loss(self, multi_domain_batch):
        """Computes the translation loss when flooding/deflooding images

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas
        """
        step_loss = 0
        self.g_opt.zero_grad()
        lambdas = self.opts.train.lambdas

        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain != "rf":
                continue

            x = batch["data"]["x"]
            m = batch["data"]["m"]
            batch_size = x.shape[0]
            z = (
                torch.empty(
                    batch_size,
                    self.opts.gen.p.latent_dim,
                    self.painter_z_h,
                    self.painter_z_w,
                )
                .normal_(mean=0, std=1.0)
                .to(self.device)
            )

            gen_flood_img = self.G.painter(z, x * (1.0 - m))

            update_loss = (
                self.losses["G"]["p"]["vgg"](
                    vgg_preprocess(gen_flood_img), vgg_preprocess(x)
                )
                * lambdas.G["p"]["vgg"]
            )

            self.logger.losses.generator.p.vgg = (
                update_loss.item() * lambdas.G["p"]["vgg"]
            )
            step_loss += update_loss

            update_loss = self.losses["G"]["p"]["tv"](gen_flood_img)
            self.logger.losses.generator.p.tv = update_loss.item()
            step_loss += update_loss

            update_loss = (
                self.losses["G"]["p"]["context"](
                    gen_flood_img * (1.0 - m), x * (1.0 - m)
                )
                * lambdas.G["p"]["context"]
            )

            self.logger.losses.generator.p.context = update_loss.item()
            step_loss += update_loss

            fake_d_global = self.D["p"]["global"](gen_flood_img)
            fake_d_local = self.D["p"]["local"](gen_flood_img * m)
            update_loss = (
                self.losses["G"]["p"]["gan"](fake_d_global, True)
                + self.losses["G"]["p"]["gan"](fake_d_local, True)
            ) * lambdas.G["p"]["gan"]

            self.logger.losses.generator.p.gan = update_loss.item()
            step_loss += update_loss

        return step_loss

    def update_d(self, multi_domain_batch, verbose=0):
        # ? split representational as in update_g
        # ? repr: domain-adaptation traduction
        self.d_opt.zero_grad()
        d_loss = self.get_d_loss(multi_domain_batch, verbose)
        d_loss.backward()
        self.d_opt_step()

        self.logger.losses.discriminator.total_loss = d_loss.item()
        self.log_losses(model_to_update="D", mode="train")

    def get_d_loss(self, multi_domain_batch, verbose=0):
        """Compute the discriminators' losses:

        * for each domain-specific batch:
        * encode the image
        * get the conditioning tensor if using spade
        * source domain is the data's domain, sequentially r|s then f|n
        * get the target domain accordingly
        * compute the translated image from the data
        * compute the source domain discriminator's loss on the data
        * compute the target domain discriminator's loss on the translated image

        # ? In this setting, each D[decoder][domain] is updated twice towards
        # real or fake data

        See readme's update d section for details

        Args:
            multi_domain_batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        zerotensor = torch.tensor(0).to(self.device)
        disc_loss = {
            "m": {"maskAdvent": zerotensor},
            "p": {"global": zerotensor, "local": zerotensor},
        }

        for batch_domain, batch in multi_domain_batch.items():
            x = batch["data"]["x"]
            m = batch["data"]["m"]
            z = self.G.encode(x)

            if batch_domain == "rf":
                # sample vector
                batch_size = x.shape[0]
                z_paint = (
                    torch.empty(
                        batch_size,
                        self.opts.gen.p.latent_dim,
                        self.painter_z_h,
                        self.painter_z_w,
                    )
                    .normal_(mean=0, std=1.0)
                    .to(self.device)
                )
                fake = self.G.painter(z_paint, x * (1.0 - m))
                fake_d_global = self.D["p"]["global"](fake)
                real_d_global = self.D["p"]["global"](x)
                fake_d_local = self.D["p"]["local"](fake * m)
                real_d_local = self.D["p"]["local"](x * m)

                global_loss = self.losses["D"]["default"](
                    fake_d_global, False
                ) + self.losses["D"]["default"](real_d_global, True)

                local_loss = self.losses["D"]["default"](
                    fake_d_local, False
                ) + self.losses["D"]["default"](real_d_local, True)

                disc_loss["p"]["global"] += global_loss
                disc_loss["p"]["local"] += local_loss

            else:
                if "m" in self.opts.tasks:
                    if verbose > 0:
                        print("Now training the ADVENT discriminator!")
                    z_decode = self.G.decoders["m"](z)
                    z_prime = 1 - z_decode
                    prob = torch.cat([z_decode, z_prime], dim=1)
                    prob = prob.detach()

                    if batch_domain == "r":
                        loss_main = self.losses["D"]["advent"](
                            prob.to(self.device),
                            self.target_label,
                            self.D["m"]["maskAdvent"],
                        )

                        # loss_aux = self.losses["D"]["advent"](
                        #     prob_aux.to(self.device),
                        #     self.target_label,
                        #     self.D["m"]["maskAdvent_aux"]
                        # )

                        disc_loss["m"]["maskAdvent"] += (
                            self.opts.train.lambdas.advent.adv_main * loss_main
                        )

                    elif batch_domain == "s":
                        loss_main = self.losses["D"]["advent"](
                            prob.to(self.device),
                            self.source_label,
                            self.D["m"]["maskAdvent"],
                        )

                        # loss_aux = self.losses["D"]["advent"](
                        #     prob_aux.to(self.device),
                        #     self.source_label,
                        #     self.D["m"]["maskAdvent_aux"]
                        # )

                        disc_loss["m"]["maskAdvent"] += (
                            self.opts.train.lambdas.advent.adv_main * loss_main
                        )
                    else:
                        raise Exception("Wrong Domain Input!")

        self.logger.losses.discriminator.update(
            {dom: {k: v.item() for k, v in d.items()} for dom, d in disc_loss.items()}
        )
        loss = sum(v for d in disc_loss.values() for k, v in d.items())
        return loss

    def update_c(self, multi_domain_batch):
        """
        Update the classifier using normal labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
                the trainer's loaders

        """
        self.c_opt.zero_grad()
        c_loss = self.get_classifier_loss(multi_domain_batch)
        # ? Log policy
        self.logger.losses.classifier = c_loss.item()
        c_loss.backward()
        self.c_opt_step()

    def get_classifier_loss(self, multi_domain_batch):
        """Compute the loss of the domain classifier with real labels

        Args:
            multi_domain_batch (dict): dictionnary mapping domain names to batches from
            the trainer's loaders

        Returns:
            torch.Tensor: scalar loss tensor, weighted according to opts.train.lambdas.C
        """
        loss = 0
        lambdas = self.opts.train.lambdas
        one_hot = self.opts.classifier.loss != "cross_entropy"
        for batch_domain, batch in multi_domain_batch.items():
            # We don't care about the flooded domain here
            if batch_domain == "rf":
                continue
            self.z = self.G.encode(batch["data"]["x"])
            # Forward through classifier, output classifier = (batch_size, 4)
            output_classifier = self.C(self.z)
            # Cross entropy loss (with sigmoid)
            update_loss = self.losses["C"](
                output_classifier,
                domains_to_class_tensor(batch["domain"], one_hot).to(self.device),
            )
            loss += update_loss

        return lambdas.C * loss

    def eval(self, num_threads=5, verbose=0):
        print("*******************EVALUATING***********************")

        lambdas = self.opts.train.lambdas
        for i, multi_batch_tuple in enumerate(self.val_loaders):
            # create a dictionnay (domain => batch) from tuple
            # (batch_domain_0, ..., batch_domain_i)
            # and send it to self.device
            multi_domain_batch = {
                batch["domain"][0]: self.batch_to_device(batch)
                for batch in multi_batch_tuple
            }

            # ----------------------------------------------
            # -----  Infer separately for each domain  -----
            # ----------------------------------------------
            for domain, domain_batch in multi_domain_batch.items():

                x = domain_batch["data"]["x"]
                self.z = self.G.encode(x)
                # Don't infer if domains has enough images

                if verbose > 0:
                    print(f"Inferring batch {i} domain {domain}")

                # translator = "f" if "n" in domain else "n"
                # domain_batch = slice_batch(domain_batch, remaining)
                # translated = self.G.translate_batch(domain_batch, translator)
                # Get task losses:
                task_tensors = {}
                for update_task, update_target in domain_batch["data"].items():
                    # task t (=translation) will be done in get_translation_loss
                    # task a (=adaptation) and x (=auto-encoding) will be done hereafter
                    if update_task not in {"t", "a", "x", "m"}:
                        # ? output features classifier
                        prediction = self.G.decoders[update_task](self.z)
                        task_tensors[update_task] = prediction
                        update_loss = self.losses["G"]["tasks"][update_task](
                            prediction, update_target
                        )
                        self.logger.losses.generator.task_loss[update_task][
                            domain
                        ] = update_loss.item()

                    if update_task == "m":
                        # ? output features classifier
                        prediction = self.G.decoders[update_task](self.z)
                        task_tensors[update_task] = prediction

                        # Main loss first:
                        update_loss = (
                            self.losses["G"]["tasks"][update_task]["main"](
                                prediction, update_target
                            )
                            * lambdas.G[update_task]["main"]
                        )
                        self.logger.losses.generator.task_loss[update_task]["main"][
                            domain
                        ] = update_loss.item()

                        # Then TV loss
                        update_loss = self.losses["G"]["tasks"][update_task]["tv"](
                            prediction
                        )
                        self.logger.losses.generator.task_loss[update_task]["tv"][
                            domain
                        ] = update_loss.item()
        self.log_losses(model_to_update="G", mode="val")

        if self.opts.art == "mask":
            self.log_comet_images("val", "r")
            self.log_comet_images("val", "s")
        elif self.opts.art == "paint":
            self.log_comet_images("val", "rf")
        else:
            raise ValueError("Unknown opts.art {}".format(self.opts.art))
        print("******************DONE EVALUATING*********************")

    def save(self):
        save_dir = Path(self.opts.output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = Path("latest_ckpt.pth")
        save_path = save_dir / save_path

        # Construct relevant state dicts / optims:
        # Save at least G
        save_dict = {
            "epoch": self.logger.epoch,
            "G": self.G.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "step": self.logger.global_step,
        }

        if self.C is not None and get_num_params(self.C) > 0:
            save_dict["C"] = self.C.state_dict()
            save_dict["c_opt"] = self.c_opt.state_dict()
        if self.D is not None and get_num_params(self.D) > 0:
            save_dict["D"] = self.D.state_dict()
            save_dict["d_opt"] = self.d_opt.state_dict()

        torch.save(save_dict, save_path)

    def resume(self):
        # load_path = self.get_latest_ckpt()
        load_path = Path(self.opts.output_path) / Path("checkpoints/latest_ckpt.pth")
        checkpoint = torch.load(load_path)
        print(f"Resuming model from {load_path}")
        self.G.load_state_dict(checkpoint["G"])
        self.g_opt.load_state_dict(checkpoint["g_opt"])
        self.logger.epoch = checkpoint["epoch"]
        self.logger.global_step = checkpoint["step"]
        # Round step to even number for extraGradient
        if self.logger.global_step % 2 != 0:
            self.logger.global_step += 1

        if self.C is not None and get_num_params(self.C) > 0:
            self.C.load_state_dict(checkpoint["C"])
            self.c_opt.load_state_dict(checkpoint["c_opt"])

        if self.D is not None and get_num_params(self.D) > 0:
            self.D.load_state_dict(checkpoint["D"])
            self.d_opt.load_state_dict(checkpoint["d_opt"])

    def get_latest_ckpt(self):
        load_dir = Path(self.opts.output_path) / Path("checkpoints")
        ckpts = os.listdir(str(load_dir.resolve()))
        max_epoch = 0
        max_ckpt = ""
        for ckpt in ckpts:
            ckpt = Path(ckpt)
            epoch = int(ckpt.stem.split("_")[-1])
            if epoch > max_epoch:
                max_epoch = epoch
                max_ckpt = ckpt
        return Path(self.opts.output_path) / Path("checkpoints") / max_ckpt

    def debug(self, func_name, local_vars, index=None):
        if self.verbose == 0:
            return

        if func_name == "get_representation_loss":
            if index == 0:
                print(
                    "Domain {} Task {} Pred {} Loss {}".format(
                        local_vars["batch_domain"],
                        local_vars["update_task"],
                        local_vars["prediction"].shape,
                        local_vars["update_loss"],
                    )
                )
            if index == 1:
                print(
                    "Translation reconstruction {} Loss {}".format(
                        local_vars["reconstruction"].shape, local_vars["update_loss"]
                    )
                )
            if index == 2:
                print(
                    "Adaptation reconstruction {} Loss {}".format(
                        local_vars["reconstruction"].shape, local_vars["update_loss"]
                    )
                )
            if index == 3:
                print("{}: {}".format("real", local_vars["real"].shape))
                print("{}: {}".format("z_real", local_vars["z_real"].shape))
                print("{}: {}".format("fake", local_vars["fake"].shape))

        if func_name == "get_translation_loss":
            print("{}: {}".format("real", local_vars["real"].shape))
            print("{}: {}".format("z_real", local_vars["z_real"].shape))
            print("{}: {}".format("fake", local_vars["fake"].shape))
            print("{}: {}".format("cycle", local_vars["cycle"].shape))
        # print("-------- end {} --------".format(func_name))
        print()
