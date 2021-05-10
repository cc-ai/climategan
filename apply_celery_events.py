print("\n• Imports\n")
# from ast import Bytes
import os
import logging
import time

logger = logging.getLogger('azure.storage')
logger.setLevel(logging.ERROR)

import_time = time.time()
import sys
from io import BytesIO
from PIL import Image
# from omnigan.data import is_image_file
from omnigan.tutils import normalize
from skimage.color import rgba2rgb
from skimage.transform import resize
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import OrderedDict
import concurrent.futures
import threading

from omnigan.utils import (
    Timer,
    get_git_revision_hash,
    to_128,
    # find_images,
)

import_time = time.time() - import_time

XLA = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    XLA = True
except ImportError:
    pass


def write_apply_config(out):
    command = " ".join(sys.argv)
    git_hash = get_git_revision_hash()
    with (out / "command.txt").open("w") as f:
        f.write(command)
    with (out / "hash.txt").open("w") as f:
        f.write(git_hash)


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

def download_blob_and_preprocess(container_client, path, output):
    """Download a blob and preprocess the image"""
    dld = container_client.download_blob(path)
    filestream = BytesIO()
    dld.readinto(filestream)

    image = np.array(Image.open(filestream))

    data = image if image.shape[-1] == 3 else rgba2rgb(image)
    # resize to standard input size 640 x 640
    data = resize(data, (640, 640), anti_aliasing=True)
    # normalize to -1:1
    data = (normalize(data.astype(np.float32)) - 0.5) * 2

    return (output, data)

def download_blobs_and_preprocess(container_client, paths_on_container=['input/'], output_paths=['output/']):
    """Download images from container using a thread pool"""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_images = [executor.submit(download_blob_and_preprocess, container_client, path, output) 
                        for path, output in zip(paths_on_container, output_paths)]
        for future in future_images:
            try:
                data = future.result()
            except Exception as exc:
                logging.error(exc)
            else:
                yield data


def upload_blobs(container_client, all_events, image_paths, stores):
    """Upload blobs using multithreading"""

    def daemon(output, container_client):
        """Task for the daemon threads to complete"""
        for im_path, im_data in output:
            imagefile = BytesIO()
            im_data.save(imagefile, format='JPEG', quality=80)
            container_client.upload_blob(
                str(im_path),
                imagefile.getvalue(),
                overwrite=True
                )            

    print("\n• Writing")
    with Timer(store=stores.get("write", [])):
        output_images = dict(zip(image_paths, [[]]*len(image_paths)))
        for event_type, events in all_events.items():
            for event, path in zip(events, image_paths):
                fpath = path if path[-1]=='/' else path+'/'
                im_path = fpath + f"{event_type}.jpg"
                im_data = Image.fromarray(event)
                output_images[path].append((im_path, im_data))

        for uploads in output_images.values():
            d = threading.Thread(name='daemon', target=daemon, args=(uploads, container_client))
            d.setDaemon(True)
            d.start()

def run_inference_from_trainer(trainer, container_client, paths_on_container=['input/'], output_paths=['output/'], time_inference=True, \
    flood_mask_binarization=0.5,
    half=True,
    cloudy = True,
    xla_purge_samples=-1,
    ):

    batch_size = len(paths_on_container)

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
                "data: download and pre-processing": [],
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
                "active_threads": [threading.active_count()],
            }
        )


    # --------------------------------------------
    # -----  Read data from input directory  -----
    # --------------------------------------------
    print("\n• Downloading & Pre-processing Data\n")

    with Timer(store=stores.get("data: download and pre-processing", [])):
        images = list(
            download_blobs_and_preprocess(
                container_client,
                paths_on_container,
                output_paths)
            )
    
    valid_output, data = zip(*images)
    n_batchs = len(data) // batch_size
    if len(data) % batch_size != 0:
        n_batchs += 1

    print("Found", len(paths_on_container), "images. Inferring on", len(data), "images.")

    # --------------------------------------------
    # -----  Batch-process images to events  -----
    # --------------------------------------------
    print(f"\n• Creating events on {str(trainer.device)}\n")

    all_events = []

    with Timer(store=stores.get("inference on all images", [])):
        # concatenate images in a batch batch_size x height x width x 3
        images = np.stack(data)

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

    # ----------------------------------------------
    # -----  Write events to output directory  -----
    # ----------------------------------------------
    upload_blobs(container_client, events, valid_output, stores)
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


