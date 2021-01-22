# omnigan
- [omnigan](#omnigan)
  - [Setup](#setup)
  - [Coding conventions](#coding-conventions)
    - [Resuming](#resuming)
- [⚠️ Deprecated](#️-deprecated)
  - [Current Model](#current-model)
    - [Summary](#summary)
    - [Generator](#generator)
    - [Discriminator](#discriminator)
  - [updates](#updates)
  - [interfaces](#interfaces)
    - [batches](#batches)
    - [data](#data)
      - [json files](#json-files)
    - [losses](#losses)
  - [Logging on comet](#logging-on-comet)
    - [Tests](#tests)
  - [Resources](#resources)
  - [Model Architecture](#model-architecture)
    - [Generator](#generator-1)
    - [Discriminators](#discriminators)
- [Running Experiments](#running-experiments)
  - [Comet-specific parameters](#comet-specific-parameters)
  - [Sampling parameters](#sampling-parameters)
- [Choices and Ideas](#choices-and-ideas)

## Setup

**`PyTorch >= 1.1.0`** otherwise optimizer.step() and scheduler.step() are in the wrong order ([docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate))

**pytorch==1.6** to use pytorch-xla or automatic mixed precision (`amp` branch).

Configuration files use the **YAML** syntax. If you don't know what `&` and `<<` mean, you'll have a hard time reading the files. Have a look at:

  * https://dev.to/paulasantamaria/introduction-to-yaml-125f
  * https://stackoverflow.com/questions/41063361/what-is-the-double-left-arrow-syntax-in-yaml-called-and-wheres-it-specced/41065222

**pip**

```
$ pip install comet_ml scipy opencv-python torch torchvision omegaconf==1.4.1 hydra-core==0.11.3 scikit-image imageio addict tqdm torch_optimizer
```

## Coding conventions

* Tasks
  * `x` is an input image, in [-1, 1]
  * `s` is a segmentation target with `long` classes
  * `d` is a depth map target in R, may be actually `log(depth)` or `1/depth`
  * `m` is a binary mask with 1s where water is/should be
* Domains
  * `r` is the *real* domain for the masker. Input images are real pictures of urban/suburban/rural areas
  * `s` is the *simulated* domain for the masker. Input images are taken from our Unity world
  * `rf` is the *real flooded* domain for the painter. Training images are pairs `(x, m)` of flooded scenes for which the water should be reconstructed, in the validation data input images are not flooded and we provide a manually labeled mask `m`
  * `kitti` is a special `s` domain to pre-train the masker on [Virtual Kitti 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
    * it alters the `trainer.loaders` dict to select relevant data sources from `trainer.all_loaders` in `trainer.switch_data()`. The rest of the code is identical.
* Flow
  * This describes the call stack for the trainers standard training procedure
  * `train()`
    * `run_epoch()`
      * `update_G()`
        * `zero_grad(G)`
        * `get_G_loss()`
          * `get_masker_loss()`
            * `masker_m_loss()`  -> masking loss
            * `masker_s_loss()`  -> segmentation loss
            * `masker_d_loss()`  -> depth estimation loss
          * `get_painter_loss()` -> painter's loss
        * `g_loss.backward()`
        * `g_opt_step()`
      * `update_D()`
        * `zero_grad(D)`
        * `get_D_loss()`
          * painter's disc losses
          * `masker_m_loss()` -> masking AdvEnt disc loss
          * `masker_s_loss()` -> segmentation AdvEnt disc loss
        * `d_loss.backward()`
        * `d_opt_step()`
      * `update_learning_rates()` -> update learning rates according to schedules defined in `opts.gen.opt` and `opts.dis.opt`
    * `run_validation()`
      * compute val losses
      * `eval_images()` -> compute metrics
      * `log_comet_images()` -> compute and upload inferences
    * `save()`

### Resuming

Use a config's `load_path` namespace. It should have sub-keys `m`, `p` and `pm`:

```yaml
load_paths:
  p: none # Painter weights
  m: none # Masker weights
  pm: none # Painter + Masker weights (single ckpt for both)
```

1. any path which leads to a dir will be loaded as `path / checkpoints / latest_ckpt.pth`
2. if you want to specify a specific checkpoint (not the latest), it MUST be a `.pth` file
3. resuming a `P` **OR** an `M` model, you may only specify 1 of `load_path.p` **OR** `load_path.m`.
   You may also leave **BOTH** at `none`, in which case `output_path / checkpoints / latest_ckpt.pth`
   will be used
4. resuming a P+M model, you may specify (`p` AND `m`) **OR** `pm` **OR** leave all at `none`,
   in which case `output_path / checkpoints / latest_ckpt.pth` will be used to load from
   a single checkpoint

# ⚠️ Deprecated

Most of the following sections need an update

## Current Model

### Summary

Summary from `torchsummary` with only 1 ResBlock in the encoder and 1 in the decoders:

```
================================================================
Total params: 9,766,807
Trainable params: 9,766,807
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 1746.82
Params size (MB): 37.26
Estimated Total Size (MB): 1784.82
----------------------------------------------------------------
```



Set `test_summary` to `True` in `tests/test_gen.py` to view the full summary.

**n.b.**: the adaptation decoder is not taken into account in the summary as its computations are not used in `OmniGenerator.forward(...)` and only one translation decoder is used so numbers above are a lower bound.

Also:

```python
sum(np.prod(p.shape) for p in trainer.G.parameters())
>>> 17 646 624

sum(np.prod(p.shape) for p in trainer.D.parameters())
>>> 11 124 424

sum(np.prod(p.shape) for p in trainer.C.parameters())
>>> 595 332
```

### Generator

High-level model in `generator.py`, building-blocks in `blocks.py`

* **Encoder**:

    Resnet-based Content Encoder from MUNIT
  * image => 64 (=`encoder.dim`) channels with 1 conv layer, same size
  * conv-based downsamplings (`encoder.n_downsample` times)
  * resblocks (`encoder.n_res` blocks)

  Deeplabv2-based encoder
  * Code borrowed from https://github.com/valeoai/ADVENT/blob/master/advent/model/deeplabv2.py
  * We only keep the feature extractor part (not the ASPP classification module) for which we can load pretrained weights.
  Pretrained model weights on ImageNet can be downloaded [here](https://github.com/valeoai/ADVENT/releases).
  * We also add resblocks

* **Decoders**: Resnet-based Decoders from MUNIT for all tasks but the translation
  * resblocks projections (`decoder.n_res` blocks)
  * Sequence of `nn.Upsampling > Conv2dBlock` (`decoder.n_upsample` times)
    * should match `encoder.n_downsample`
  * final conv to get a feature map with 1 (`h`, `d`, `w`), 3 (`a`) or 19 (`s`) channels
* **Translation decoder**: SPADEResnet-based Decoder inspired by MUNIT and SPADE
  * Conditioning the translation by `SPADE([h, d, s, w])`

### Discriminator

## updates

multi-batch:

```
multi_domain_batch = {"rf: batch0, "r": batch1, "s": batch2}
```

## interfaces

### batches
```python
batch = Dict({
    "data": {
        "d": depthmap,,
        "s": segmentation_map,
        "m": binary_mask
        "x": real_flooded_image,
    },
    "paths":{
        same_keys: path_to_file
    }
    "domain": list(rf | r | s),
    "mode": list(train | val)
})
```

### data

#### json files

| name                                           | domain | description                                                                |  author   |
| :--------------------------------------------- | :----: | :------------------------------------------------------------------------- | :-------: |
| **train_r_full.json, val_r_full.json**         |   r    | MiDaS+ Segmentation pseudo-labels .pt (HRNet + Cityscapes)                 | Mélisande |
| **train_s_full.json, val_s_full.json**         |   s    | Simulated data from Unity11k urban + Unity suburban dataset                |    ***    |
| train_s_nofences.json, val_s_nofences.json     |   s    | Simulated data from Unity11k urban + Unity suburban dataset without fences |  Alexia   |
| train_r_full_pl.json, val_r_full_pl.json       |   r    | MegaDepth + Segmentation pseudo-labels .pt (HRNet + Cityscapes)            |  Alexia   |
| train_r_full_midas.json, val_r_full_midas.json |   r    | MiDaS+ Segmentation (HRNet + Cityscapes)                                   | Mélisande |
| train_r_full_old.json, val_r_full_old.json     |   r    | MegaDepth+ Segmentation (HRNet + Cityscapes)                               |    ***    |
| train_r_nopeople.json, val_r_nopeople.json     |   r    | Same training data as above with people removed                            |   Sasha   |
| train_rf_with_sim.json                         |   rf   | Doubled train_rf's size with sim data  (randomly chosen)                   |  Victor   |
| train_rf.json                                  |   rf   | UPDATE (12/12/20): added 50 ims & masks from ADE20K Outdoors               |  Victor   |

We provide the script `process_data.py` for the preprocessing task. Given a source folder the script will create the appropriate JSON. In the default mode, only one JSON for the whole data folder will be created. If you want to split the dataset into train and validation you can use the `--train_size` argument and specify the percentage, therefore two JSONs (train and val) will be created.

The data folder must be structured as follows : A specific folder for each category (Semantic, Depth, Height, Flood, etc..), for the same data sample, the name must be the same through all the directories (The ground truth depth of Flood/image_1.jpg is Depth/image_1.jpg), but the extension can change.

The default mapping between a folder and the json key is `"Segmentation": "s", "Depth": "d", "Data": "x", "Height": "h", "Mask": "m"` change it according to your needs in `process_data.py`.
Note that `"m"` corresponds to the mask task, which is not exactly the water mask since for real images we do not have access to that information.

```yaml
# data file ; one for each r|s
- x: /path/to/image
  m: /path/to/mask
  s: /path/to/segmentation map
- x: /path/to/another image
  d: /path/to/depth map
  m: /path/to/mask
  s: /path/to/segmentation map
- x: ...
```

or

```json
[
    {
        "x": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005.jpg",
        "s": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005.npy",
        "d": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005_depth.jpg"
    },
    {
        "x": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006.jpg",
        "s": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006.npy",
        "d": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006_depth.jpg"
    }
]
```
The json files used are located at `/network/tmp1/ccai/data/omnigan/`. In the basenames,  `_s` denotes simulated domain data and `_r` real domain data.
The `base` folder contains json files with paths to images (`"x"`key) and masks (taken as ground truth for the area that should be flooded, `"m"` key).
The `seg` folder contains json files and keys `"x"`, `"m"` and `"s"` (segmentation) for each image.


loaders

```
loaders = Dict({
    train: { r: loader, s: loader},
    val: { r: loader, s: loader}
})
```

### losses

`trainer.losses` is a dictionary mapping to loss functions to optimize for the 3 main parts of the architecture: generator `G`, discriminators `D`, domain classifier `C`:

```python
trainer.losses = {
    "G":{ # generator
        "gan": { # gan loss from the discriminators
            "a": GANLoss, # adaptation decoder
            "t": GANLoss # translation decoder
        },
        "cycle": { # cycle-consistency loss
            "a": l1 | l2,,
            "t": l1 | l2,
        },
        "auto": { # auto-encoding loss a.k.a. reconstruction loss
            "a": l1 | l2,
            "t": l1 | l2
        },
        "tasks": {  # specific losses for each auxillary task
            "d": func, # depth estimation
            "h": func, # height estimation
            "s": cross_entropy_2d, # segmentation
            "w": func, # water generation
        },
        "classifier": l1 | l2 | CE # loss from fooling the classifier
    },
    "D": GANLoss, # discriminator losses from the generator and true data
    "C": l1 | l2 | CE # classifier should predict the right 1-h vector [rf, rn, sf, sn]
}
```

## Logging on comet

Comet.ml will look for api keys in the following order: argument to the `Experiment(api_key=...)` call, `COMET_API_KEY` environment variable, `.comet.config` file in the current working directory, `.comet.config` in the current user's home directory.

If your not managing several comet accounts at the same time, I recommend putting `.comet.config` in your home as such:

```
[comet]
api_key=<api_key>
workspace=vict0rsch
rest_api_key=<rest_api_key>
```

### Tests

Run tests by executing `python test_trainer.py`. You can add `--no_delete` not to delete the comet experiment at exit and inspect uploads.

Write tests as scenarios by adding to the list `test_scenarios` in the file. A scenario is a dict of overrides over the base opts in `shared/trainer/defaults.yaml`. You can create special flags for the scenario by adding keys which start with `__`. For instance, `__doc` is a mandatory key in any scenario describing it succinctly.

## Resources

[Tricks and Tips for Training a GAN](https://chloes-dl.com/2019/11/19/tricks-and-tips-for-training-a-gan/)
[GAN Hacks](https://github.com/soumith/ganhacks)
[Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)

## Model Architecture


### Generator

* Encoder
  *  Projection Conv layer from `3 x h x w` to `dim x h x w`
  *  `n_downsample` layers, each doubling `dim` and halving `h` and `w`
  *  `n_res` ResBlocks

* Translation decoder
  * For each `spade_n_up` which is equal to `n_downsample`: `SPADEResnetBlock` then `nn.Upsample(scale_factor=2)`
    * First 2 SRB keep as many channels as `z` (:=`z_nc`)
    * Each subsequent one halves `z_nc`
  * Image layer projects the last SRB's output to 3 channels
    * `tanh(conv(LeakyRelu(y)))`

### Discriminators

TODO

# Running Experiments

Using `experiment.py` you can easily run `sbatch` jobs to run trainings. It takes two arguments:

* **`--experiment`**: path to the experiment YAML configuration file
* **`--template`**: path to the sbatch template to run

`python experiment.py --experiment=shared/experiment/exp-d.yaml --template=shared/experiment/template.sh` will:

1. load `exp-d.yaml` as an addict.Dict `xopts`
   1. load `xopts.defaults` as default parameters for the trainers: they are those shared across all experiments
   2. load `xopts.config` to override `xopts.defaults` for the trainers: they are those shared across trainers for this experiment
   3. load `xopts.runs[i].trainer` to override `xopts.config`: these are trainer `i`'s specific parameters within the experiment
2. create an `sbatch` file `exp.sh` from `--template`, filling `{{param}}` according to `omnigan.utils.write_run_template`, by specifying `cpu`, `gpu`, args to `train.py` etc.
3. write reproducibility files:
   1. git hash code to `hash.txt`
   2. comet url to `comet_url.txt`
   3. experiment file to `exp_i.yaml` where `i` is the index of the run in `xopts.runs`
   4. trainer config file to `config.yaml`
   5. `sbatch` launch file in `exp.sh`

## Comet-specific parameters

* `experiment.exp_desc`: Overall description of the experiment
* `runs[i].comet.note`: Text description of this run
* `runs[i].comet.tags`: tags for the comet experiment log

## Sampling parameters

In an **experiment** config file, you may switch a parameter's value with a dict to sample this parameter from some distribution instead of just specifying it:

```yaml
runs:
  - trainer:
    gen:
      opt:
        lr:
          sample: list            | range            | uniform
          from: "[list of values] | [min, max, step] | [low, high, size]"

```

# Choices and Ideas

* Implementation relies on the idea of splitting the representation and translation training phases
  * see Samuel Lavoie's work
* This allows the task decoders to be already trained when used in translation (to condition the generation and add consistency losses (depth, semantic, height))
* When the Translation phase starts, representation blocks (encoder + task decoders) could be
  * frozen
  * trainable
    * use Continual Learning ideas to prevent forgetting
    * greatly lower learning rate
