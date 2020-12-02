# omnigan
- [omnigan](#omnigan)
  - [Setup](#setup)
  - [Coding conventions](#coding-conventions)
    - [Resuming](#resuming)
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
      * `update_learning_rates()` -> update learning rates according to schedules defined in `otps.gen.opt` and `opts.dis.opt`
    * `run_validation()`
      * compute val losses
      * `eval_images()` -> compute metrics
      * `log_comet_images()` -> compute and upload inferences
    * `save()`

### Resuming

Set  `train.resume` to `True` in `opts.yaml` and specify where to load the weights:

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

### Generator

* **Encoder**:

  `trainer.G.encoder` Deeplabv2 or v3-based encoder
  * Code borrowed from
    * https://github.com/valeoai/ADVENT/blob/master/advent/model/deeplabv2.py
    * https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes

* **Decoders**:
  * `trainer.G.decoders["s"]` -> *Segmentation* -> DLV3+ architecture (ASPP + Decoder)
  * `trainer.G.decoders["d"]` -> *Depth* -> ResBlocks + (Upsample + Conv)
  * `trainer.G.decoders["m"]` -> *Mask* -> ResBlocks + (Upsample + Conv) -> Binary mask: 1 = water should be there
    * `trainer.G.mask()` predicts a mask and optionally applies `sigmoid` from an `x` input or a `z` input

* **Painter**: `trainer.G.painter` -> [GauGAN SPADE-based](https://github.com/NVlabs/SPADE)
  * input = masked image
* `trainer.G.paint(m, x)` higher level function which takes care of masking
* If `opts.gen.p.paste_original_content` the painter should only create water and not reconstruct outside the mask: the output of `paint()` is `painted * m + x * (1 - m)`

High level methods of interest:

* `trainer.infer_all()` creates a dictionary of events with keys `flood` `wildfire` and `smog`. Can take in a single image or a batch, of numpy arrays or torch tensors, on CPU/GPU/TPU. This method calls, amongst others:
  * `trainer.G.encode()` to compute the shared latent vector `z`
  * `trainer.G.mask(z=z)` to infer the mask
  * `trainer.compute_fire(x, segmentation)` to create a wildfire image from `x` and inferred segmentation
  * `trainer.compute_smog(x, depth)` to create a smog image from `x` and inferred depth
  * `trainer.compute_flood(x, mask)` to create a flood image from `x` and inferred mask using the painter (`trainer.G.paint(m, x)`)
* `Trainer.resume_from_path()` static method to resume a trainer from a path

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

`trainer.losses` is a dictionary mapping to loss functions to optimize for the 3 main parts of the architecture: generator `G`, discriminators `D`:

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

There's a `test_comet.py` test which will automatically start and stop an experiment, check that logging works and so on. Not to pollute your workspace, such functional tests are deleted when the test is passed through Comet's REST API which is why you need to specify this `rest_api_key` field.

Set `should_delete` to False in the file not to delete the test experiment once it has ended. You'll be able to find all your test experiments which were not deleted using the `is_functional_test` parameter on Comet's web interface.

## Resources

[Tricks and Tips for Training a GAN](https://chloes-dl.com/2019/11/19/tricks-and-tips-for-training-a-gan/)
[GAN Hacks](https://github.com/soumith/ganhacks)
[Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)
