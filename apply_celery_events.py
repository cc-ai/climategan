print("\n• Imports\n")
from ast import Bytes
import logging
import time

import_time = time.time()
import os
from io import BytesIO
from PIL import Image
from omnigan.data import is_image_file
from omnigan.utils import Timer
from omnigan.tutils import normalize
import skimage.io as io
from skimage.transform import resize
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import OrderedDict

from azure.storage.blob import BlobServiceClient


import_time = time.time() - import_time

XLA = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    XLA = True
except ImportError:
    pass


def print_time(text, time_series, purge=-1):
    """
    Print a timeseries's mean and std with a label

    Args:
        text (str): label of the time series
        time_series (list): list of timings
        purge (int, optional): ignore first n values of time series. Defaults to -1.
    """
    if not time_series:
        return

    if purge > 0 and len(time_series) > purge:
        time_series = time_series[purge:]

    m = np.mean(time_series)
    s = np.std(time_series)

    print(
        f"{text.capitalize() + ' ':.<26}  {m:.5f}"
        + (f" +/- {s:.5f}" if len(time_series) > 1 else "")
    )


def print_store(store, purge=-1):
    """
    Pretty-print time series store

    Args:
        store (dict): maps string keys to lists of times
        purge (int, optional): ignore first n values of time series. Defaults to -1.
    """
    singles = OrderedDict({k: v for k, v in store.items() if len(v) == 1})
    multiples = OrderedDict({k: v for k, v in store.items() if len(v) > 1})
    empties = {k: v for k, v in store.items() if len(v) == 0}

    if empties:
        print("Ignoring empty stores ", ", ".join(empties.keys()))
        print()

    for k in singles:
        print_time(k, singles[k], purge)

    print()
    print("Unit: s/batch")
    for k in multiples:
        print_time(k, multiples[k], purge)
    print()


def connect_to_container(container):
    stringauth = 'DefaultEndpointsProtocol=https;AccountName=ccviz;AccountKey=RWl7PcQoR3HqeEVr9M1ECCg6AIc8X1RlpqjF1TgsDbZjFqMZxY4heBpk5x3WvNSyT8iSfQJSx7lAKoYj4Cl8Bg==;EndpointSuffix=core.windows.net'
    return BlobServiceClient.from_connection_string(stringauth).get_container_client(container)

def download_blobs(container_client, path_on_container='input/'):
    """
    Download images from container
    """
    for blob in container_client.list_blobs(name_starts_with=path_on_container):
        dld = container_client.download_blob(blob)
        filestream = BytesIO()
        dld.readinto(filestream)
        yield Path(blob.name).stem, np.array(Image.open(filestream))


def run_inference_from_trainer(trainer, container, path_on_container='input/', output_path='output/', time_inference=True, \
    batch_size=8,
    flood_mask_binarization=-1,
    half=False,
    n_images=-1,
    xla_purge_samples=-1
    ):

    # print(
    #     "• Using args\n\n"
    #     + "\n".join(["{:25}: {}".format(k, v) for k, v in vars(kwargs).items()]),
    # )

    container_client = connect_to_container(container)
    bin_value = flood_mask_binarization

    # -------------------------------
    # -----  Create time store  -----
    # -------------------------------
    stores = {}
    if time_inference:
        stores = OrderedDict(
            {
                "imports": [import_time],
                "setup": [],
                "data 1: download": [],
                "data 2: resize": [],
                "data 3: normalize": [],
                "data 4: full pre-processing": [],
                "encode": [],
                "mask": [],
                "flood": [],
                "depth": [],
                "segmentation": [],
                "smog": [],
                "wildfire": [],
                "all events": [],
                "numpy": [],
                "inference on all images": [],
                "write": [],
            }
        )

    # -------------------------------------
    # -----  Resume Trainer instance  -----
    # -------------------------------------
    print("\n• Initializing trainer\n")

    with Timer(store=stores.get("setup", [])):

        if half:
            trainer.G.half()

    # --------------------------------------------
    # -----  Read data from input directory  -----
    # --------------------------------------------
    print("\n• Downloading & Pre-processing Data\n")

    with Timer(store=stores.get("data 4: full pre-processing", [])):
        with Timer(store=stores.get("data 1: download", [])):
            images = list(download_blobs(container_client, path_on_container))
        base_images = images
        n_image_streams = len(base_images)
        # filter images
        nstreams = len(images)
        if 0 < n_images < nstreams:
            images = images[:n_images]
        # repeat data
        elif n_images > nstreams:
            repeats = n_images // nstreams + 1
            images = base_images * repeats
            images = images[:n_images]

        image_names, data = zip(*images)
        with Timer(store=stores.get("data 2: resize", [])):
            # resize to standard input size 640 x 640
            data = [resize(d, (640, 640), anti_aliasing=True) for d in data]
        with Timer(store=stores.get("data 3: normalize", [])):
            # normalize to -1:1
            data = [(normalize(d.astype(np.float32)) - 0.5) * 2 for d in data]

    n_batchs = len(data) // batch_size
    if len(data) % batch_size != 0:
        n_batchs += 1

    print("Found", n_image_streams, "images. Inferring on", len(data), "images.")

    # --------------------------------------------
    # -----  Batch-process images to events  -----
    # --------------------------------------------
    print(f"\n• Creating events on {str(trainer.device)}\n")

    all_events = []

    with Timer(store=stores.get("inference on all images", [])):
        for b in range(n_batchs):
            print(f"Batch {b + 1}/{n_batchs}", end="\r")
            images = data[b * batch_size : (b + 1) * batch_size]
            if not images:
                continue
            # concatenate images in a batch batch_size x height x width x 3
            images = np.stack(images)

            # Retreive numpy events as a dict {event: array}
            events = trainer.infer_all(
                images,
                numpy=True,
                stores=stores,
                bin_value=bin_value,
                half=half,
                xla=XLA,
                cloudy=cloudy,
            )

            # store events to write after inference loop
            all_events.append(events)
    print()

    # ----------------------------------------------
    # -----  Write events to output directory  -----
    # ----------------------------------------------
    if output_path is not None:
        print("\n• Writing")
        with Timer(store=stores.get("write", [])):
            for b, events in enumerate(all_events):
                for i in range(len(list(events.values())[0])):
                    idx = b * batch_size + i
                    idx = idx % n_image_streams
                    stem = image_names[idx]
                    for event in events:
                        im_path = output_path + f"{stem}_{event}.png"
                        im_data = Image.fromarray(events[event][i])
                        imagefile = BytesIO()
                        im_data.save(imagefile, format='PNG')
                        container_client.upload_blob(str(im_path), imagefile.getvalue(), overwrite=True)

    # ---------------------------
    # -----  Print timings  -----
    # ---------------------------
    if time_inference:
        print("\n• Timings\n")
        print_store(stores, purge=xla_purge_samples)

    if XLA:
        metrics_dir = Path(__file__).parent / "config" / "metrics"
        metrics_dir.mkdir(exist_ok=True, parents=True)
        now = str(datetime.now()).replace(" ", "_")
        with open(metrics_dir / f"xla_metrics_{now}.txt", "w",) as f:
            report = met.metrics_report()
            print(report, file=f)
