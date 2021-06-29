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

1. Download and unzip the weights [from this link]()
2. Put them in `config/`
    
    1. With `comet`:

    ```bash
    python apply_events.py --batch_size 4 --half --images_paths path/to/a/folder --resume_path config/model/masker --upload
    ```

    2. Without `comet` (and shortened args compared to the previous example):

    ```bash
    python apply_events.py -b 4 --half -i path/to/a/folder -r config/model/masker --output_path path/to/a/folder
    ```