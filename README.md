# omnigan
- [omnigan](#omnigan)
  - [Setup](#setup)
  - [Current Model](#current-model)
    - [Summary](#summary)
    - [Generator](#generator)
    - [Discriminator](#discriminator)
  - [updates](#updates)
  - [interfaces](#interfaces)
    - [batches](#batches)
    - [data](#data)
    - [losses](#losses)
  - [Logging on comet](#logging-on-comet)
    - [Parameters](#parameters)
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

Configuration files use the **YAML** syntax. If you don't know what `&` and `<<` mean, you'll have a hard time reading the files. Have a look at:

    * https://dev.to/paulasantamaria/introduction-to-yaml-125f
    * https://stackoverflow.com/questions/41063361/what-is-the-double-left-arrow-syntax-in-yaml-called-and-wheres-it-specced/41065222

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

For a multi-batch like:
```
{"rf: batch0, "rn": batch1, "sf": batch2 "sn": batch3}
```

Updates will be:

```
- real D["a"]["r"] (batch0)
- fake D["a"]["s"] (batch0)
- real D["t"]["f"] (batch0)
- fake D["t"]["n"] (batch0)

- real D["a"]["r"] (batch1)
- fake D["a"]["s"] (batch1)
- real D["t"]["n"] (batch1)
- fake D["t"]["f"] (batch1)

- real D["a"]["s"] (batch2)
- fake D["a"]["r"] (batch2)
- real D["t"]["f"] (batch2)
- fake D["t"]["n"] (batch2)

- real D["a"]["s"] (batch3)
- fake D["a"]["r"] (batch3)
- real D["t"]["n"] (batch3)
- fake D["t"]["f"] (batch3)
```

## interfaces

### batches
```python
batch = Dict({
    "data": {
        "d": depthmap,
        "h": heightmap,
        "w": water_segmentation_map,
        "s": segmentation_map,
        "x": real_flooded_image,
    },
    "paths":{
        "d": depthmap_path,
        "h": heightmap_path,
        "w": water_segmentation_map_path,
        "s": segmentation_map_path,
        "x": real_flooded_image_path,
    }
    "domain": list(rf | rn | sf | sn),
    "mode": list(train | val)
})
```

### data

We provide the script `process_data.py` for the preprocessing task. Given a source folder the script will create the appropriate JSON. In the default mode, only one JSON for the whole data folder will be created. If you want to split the dataset into train and validation you can use the `--train_size` argument and specify the percentage, therefore two JSONs (train and val) will be created.

The data folder must be structured as follows : A specific folder for each category (Semantic, Depth, Height, Flood, etc..), for the same data sample, the name must be the same through all the directories (The ground truth depth of Flood/image_1.jpg is Depth/image_1.jpg), but the extension can change.

The default mapping between a folder and the json key is `"Segmentation": "s", "Depth": "d", "Data": "x", "Height": "h"` change it according to your needs in `process_data.py`.


```yaml
# data file ; one for each r|s-f|n
- x: /path/to/image
  h: /path/to/height map
  d: /path/to/depth map
  w: /path/to/water map
  s: /path/to/segmentation map
- x: /path/to/another image
  d: /path/to/depth map
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

loaders

```
loaders = Dict({
    train: { rn: loader, rf: loader, sn: loader, sf: loader},
    val: { rn: loader, rf: loader, sn: loader, sf: loader}
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

### Parameters

Set `train.log_level` in your configuration file to control the amount of logging on comet:

* `0`: no logging on comet
* `1`: only aggregated losses (representational loss, translation loss, total loss)
* `2`: all losses (aggregated + task losses + auto-encoding losses)

### Tests

There's a `test_comet.py` test which will automatically start and stop an experiment, check that logging works and so on. Not to pollute your workspace, such functional tests are deleted when the test is passed through Comet's REST API which is why you need to specify this `rest_api_key` field.

Set `should_delete` to False in the file not to delete the test experiment once it has ended. You'll be able to find all your test experiments which were not deleted using the `is_functional_test` parameter on Comet's web interface.

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
