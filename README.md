# ClimateGAN: Raising Awareness about Climate Change by Generating Images of Floods

This repository contains the code used to train the model presented in our **[paper]()**.

It is not simply a presentation repository but the code we have used over the past 30 months to come to our final architecture. As such, you will find many scripts, classes, blocks and options which we actively use for our own development purposes but are not directly relevant to reproduce results or use pretrained weights.

![flood processing](images/flood.png)

## Using pre-trained weights

In the paper, we present ClimateGAN as a solution to produce images of floods. It can actually do **more**: 

* reusing the segmentation map, we are able to isolate the sky, turn it red and in a few more steps create an image resembling the consequences of a wildfire on a neighboring area, similarly to the [California wildfires](https://www.google.com/search?q=california+wildfires+red+sky&source=lnms&tbm=isch&sa=X&ved=2ahUKEwisws-hx7zxAhXxyYUKHQyKBUwQ_AUoAXoECAEQBA&biw=1680&bih=917&dpr=2).
* reusing the depth map, we can simulate the consequences of a smog event on an image, scaling the intensity of the filter by the distance of an object to the camera, as per [HazeRD](http://www2.ece.rochester.edu/~gsharma/papers/Zhang_ICIP2017_HazeRD.pdf)

![image of wildfire processing](images/wildfire.png)
![image of smog processing](images/smog.png)

In this section we'll explain how to produce the `Painted Input` along with the Smog and Wildfire outputs of a pre-trained ClimateGAN model.

### Installation

```bash
$ git clone git@github.com/cc-ai/climategan.git
$ pip install -r requirements-3.8.2.txt
```

Our pipeline uses [comet.ml](https://comet.ml) to log images. You don't *have* to use their services but we recommend you do as images can be uploaded on your workspace instead of being written to disk.

If you want to use Comet, make sure you have the [appropriate configuration in place (API key and workspace at least)](https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup)

### Inference

1. Download and unzip the weights [from this link](https://drive.google.com/file/d/1nAXs6injZS5pMohlwWRWrM8ORdgM79BS/view?usp=sharing)
2. Put them in `config/`
    
    1. With `comet`:

    ```bash
    python apply_events.py --batch_size 4 --half --images_paths path/to/a/folder --resume_path config/model/masker --upload
    ```

    2. Without `comet` (and shortened args compared to the previous example):

    ```bash
    python apply_events.py -b 4 --half -i path/to/a/folder -r config/model/masker --output_path path/to/a/folder
    ```

## Training from scratch

ClimateGAN is split in two main components: the Masker producing a binary mask of where water should go and the Painter generating water within this mask given an initial image's context.

### Configuration

The code is structured to use `shared/trainer/defaults.yaml` as default configuration. There are 2 ways of overriding those for your purposes (without altering that file):

1. By providing an alternative configuration as command line argument `config=path/to/config.yaml`

    1. The code will first load `shared/trainer/defaults.yaml`
    2. *then* update the resulting dictionary with values read in the provided `config` argument.
    3. The folder `config/` is NOT tracked by git so you would typically put them there

2. By overwriting specific arguments from the command-line like `python train.py data.loaders.batch_size=8`


### Data

#### Masker

##### Real Images

Because of copyrights issues we are not able to share the real images scrapped from the internet. You would have to do that yourself. In the `yaml` config file, the code expects a key pointing to a `json` file like `data.files.<train or val>.r: <path/to/a/json/file>`. This `json` file should be a list of dictionaries with tasks as keys and files as values. Example:

```json
[
    {
    "x": "path/to/a/real/image",
    "s": "path/to/a/segmentation_map",
    "d": "path/to/a/depth_map"
    },
...
]
```

Following the [ADVENT](https://github.com/valeoai/ADVENT) procedure, only `x` should be required. We use `s` and `d` inferred from pre-trained models (DeepLab v3+ and MiDAS) to use those pseudo-labels in the first epochs of training (see `pseudo:` in the config file)

##### Simulated Images

We share snapshots of the Virtual World we created in the [Mila-Simulated-Flood dataset](). You can download and unzip one water-level and then produce json files similar to that of the real data, with an additional key `"m": "path/to/a/ground_truth_sim_mask"`. Lastly, edit the config file: `data.files.<train or val>.s: <path/to/a/json/file>`

#### Painter

The painter expects input images and binary masks to train using the [GauGAN](https://github.com/NVlabs/SPADE) training procedure. Unfortunately we cannot share openly the collected data, but similarly as for the Masker's real data you would point to the data using a `json` file as:

```json
[
    {
    "x": "path/to/a/real/image",
    "m": "path/to/a/water_mask",
    },
...
]
```

And put those files as values to `data.files.<train or val>.rf: <path/to/a/json/file>` in the configuration.