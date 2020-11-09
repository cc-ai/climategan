from comet_ml import Experiment
import torch
import omnigan

if __name__ == "__main__":

    opts = omnigan.utils.load_opts()
    opts.data.check_samples = False
    opts.train.fid.n_images = 5
    opts.comet.display_size = 5
    opts.tasks = ["m", "s", "d"]
    opts.domains = ["r", "s"]
    opts.data.loaders.num_workers = 4
    opts.data.loaders.batch_size = 3
    opts.data.max_samples = 9
    opts.train.epochs = 2

    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=None,)
    trainer.setup()
    trainer.train()

    del trainer
    torch.cuda.empty_cache()

    trainer = omnigan.trainer.Trainer(opts=opts, comet_exp=Experiment())
    trainer.setup()
    trainer.train()
