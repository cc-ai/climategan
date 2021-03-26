print("\n• Imports\n")
# from ast import Bytes
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
from multiprocessing import cpu_count
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


def download_blobs(container_client, single=True, path_on_container='input/'):
    """
    Download images from container
    """
    if single:
        dld = container_client.download_blob(path_on_container)
        filestream = BytesIO()
        dld.readinto(filestream)
        yield from [['original', np.array(Image.open(filestream))]]

    else:
        for blob in container_client.list_blobs(name_starts_with=path_on_container):
            dld = container_client.download_blob(blob, max_concurrency=max_concurrency)
            filestream = BytesIO()
            dld.readinto(filestream)
            yield Path(blob.name).stem, np.array(Image.open(filestream))

# def download_blobs(container_client, max_concurrency, path_on_container='input/'):
#     """
#     Download images from container
#     """
#     def dld_and_yield(blob):
#         dld = container_client.download_blob(blob, max_concurrency=max_concurrency)
#         filestream = BytesIO()
#         dld.readinto(filestream)
#         return Path(blob.name).stem, np.array(Image.open(filestream))


#     blobs = container_client.list_blobs(name_starts_with=path_on_container)
    
#     with Pool(max_concurrency) as p:
#         return p.map(dld_and_yield, blobs)

def run_inference_from_trainer(trainer, container_client, path_on_container='input/', output_path='output/', time_inference=True, \
    batch_size=1,
    flood_mask_binarization=0.5,
    half=False,
    cloudy = True,
    n_images=-1,
    xla_purge_samples=-1,
    keep_ratio_128=False,
    max_im_width=-1,
    max_io_concurrency=-1,
    single=True,
    ):

    # print(
    #     "• Using args\n\n"
    #     + "\n".join(["{:25}: {}".format(k, v) for k, v in vars(kwargs).items()]),
    # )

    if max_io_concurrency < 1:
        max_io_concurrency = min(cpu_count(), batch_size)

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
                "n_threads": [max_io_concurrency],
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

    if keep_ratio_128:
        if batch_size != 1:
            print("\nWARNING: batch_size overwritten to 1 when using keep_ratio_128")
            batch_size = 1
        if max_im_width > 0 and max_im_width % 128 != 0:
            new_im_width = int(max_im_width / 128) * 128
            print("\nWARNING: max_im_width should be <0 or a multiple of 128.")
            print(
                "            Was {} but is now overwritten to {}".format(
                    max_im_width, new_im_width
                )
            )
            max_im_width = new_im_width

    with Timer(store=stores.get("data 4: full pre-processing", [])):
        with Timer(store=stores.get("data 1: download", [])):
            images = list(
                download_blobs(
                    container_client,
                    single,
                    path_on_container)
                )
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
        data = [im if im.shape[-1] == 3 else rgba2rgb(im) for im in data]
        with Timer(store=stores.get("data 2: resize", [])):
            # resize to standard input size 640 x 640
            if keep_ratio_128:
                new_sizes = [to_128(d, max_im_width) for d in data]
                data = [resize(d, ns, anti_aliasing=True) for d, ns in zip(data, new_sizes)]
            else:
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
        output_images = []
        output_path = output_path if output_path[-1]=='/' else output_path+'/'
        print("\n• Writing")
        with Timer(store=stores.get("write", [])):
            for b, events in enumerate(all_events):
                for i in range(len(list(events.values())[0])):
                    idx = b * batch_size + i
                    idx = idx % n_image_streams
                    stem = image_names[idx]
                    for event in events:
                        im_path = output_path + f"{event}.jpg" if single else output_path + f"{stem}_{event}.jpg"
                        im_data = Image.fromarray(events[event][i])
                        output_images.append((im_path, im_data))

    d = threading.Thread(name='daemon', target=daemon, args=(output_images, container_client))
    d.setDaemon(True)
    d.start()                        

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

def daemon(output, container_client):
    print('Starting upload')
    for im_path, im_data in output:
        imagefile = BytesIO()
        im_data.save(imagefile, format='JPEG', quality=80)
        container_client.upload_blob(
            str(im_path),
            imagefile.getvalue(),
            overwrite=True
            )            
    print('Done upload')
