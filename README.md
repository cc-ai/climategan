# omnigan
- [omnigan](#omnigan)
  - [Current Model](#current-model)
    - [Summary](#summary)
    - [Generator](#generator)
  - [interfaces](#interfaces)
    - [batches](#batches)
    - [data](#data)
    - [losses](#losses)
  - [Logging on comet](#logging-on-comet)
    - [Parameters](#parameters)
    - [Tests](#tests)

## Current Model

### Summary

Extract summary from `torchsummary`:

```
================================================================
Total params: 39,114,905
Trainable params: 39,114,905
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 4251.21
Params size (MB): 149.21
Estimated Total Size (MB): 4401.17
----------------------------------------------------------------
```

Set `test_summary` to `True` in `tests/test_gen.py` to view the full summary.

**n.b.**: the adaptation decoder is not taken into account in the summary as its computations are not used in `OmniGenerator.forward(...)` and only one translation decoder is used so numbers above are a lower bound.

### Generator

High-level model in `generator.py`, building-blocks in `blocks.py`

* **Encoder**: Resnet-based Content Encoder from MUNIT
  * image => 64 (=`encoder.dim`) channels with 1 conv layer, same size
  * conv-based downsamplings (`encoder.n_downsample` times)
  * resblocks (`encoder.n_res` blocks)
* **Decoders**: Resnet-based Decoders from MUNIT for all tasks but the translation
  * resblocks projections (`decoder.n_res` blocks)
  * Sequence of `nn.Upsampling > Conv2dBlock` (`decoder.n_upsample` times)
    * should match `encoder.n_downsample`
  * final conv to get a feature map with 1 (`h`, `d`, `w`), 3 (`a`) or 19 (`s`) channels
* **Translation decoder**: SPADEResnet-based Decoder inspired by MUNIT and SPADE
  * Conditioning the translation by `SPADE([h, d, s, w])`

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
    "domain": rf | rn | sf | sn,
    "mode": train | val
})
```

### data

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
            "a": func, # adaptation decoder
            "t": func # translation decoder
        },
        "cycle": { # cycle-consistency loss
            "a": func,
            "t": func
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
    "D":{}, # discriminator losses from the generator and true data
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
