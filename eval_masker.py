"""
Compute metrics of the performance of the masker using a set of ground-truth labels

run eval_masker.py --model "/miniscratch/_groups/ccai/checkpoints/masker/victor/no_spade/msd (17)"

"""
print("Imports...", end="")
import os.path
import os
from argparse import ArgumentParser
from pathlib import Path

from comet_ml import Experiment

import numpy as np
import pandas as pd
from skimage.color import rgba2rgb

import torch

from omnigan.data import encode_mask_label
from omnigan.utils import find_images
from omnigan.trainer import Trainer
from omnigan.transforms import PrepareTest
from omnigan.eval_metrics import pred_cannot, missed_must, may_flood, masker_metrics

print("Ok.")


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--model", required=True, type=str, help="Path to a pre-trained model",
    )
    parser.add_argument(
        "--images_dir",
        default="/miniscratch/_groups/ccai/data/floodmasks_eval/imgs",
        type=str,
        help="Directory containing the original test images",
    )
    parser.add_argument(
        "--labels_dir",
        default="/miniscratch/_groups/ccai/data/floodmasks_eval/labels",
        type=str,
        help="Directory containing the labeled images",
    )
    parser.add_argument(
        "--preds_dir",
        default="/miniscratch/_groups/ccai/data/omnigan/flood_eval_inferred_masks",
        type=str,
        help="DEBUG: Directory containing pre-computed mask predictions",
    )
    parser.add_argument(
        "--image_size",
        default=640,
        type=int,
        help="The height and weight of the pre-processed images",
    )
    parser.add_argument(
        "--limit", default=-1, type=int, help="Limit loaded samples",
    )
    parser.add_argument(
        "--bin_value", default=-1, type=float, help="Mask binarization threshold"
    )

    return parser.parse_args()


def plot_labels_images(*args, **kwargs):
    return []


def get_inferences(image_arrays, model_path, verbose=0):
    """
    Obtains the mask predictions of a model for a set of images

    Parameters
    ----------
    image_arrays : array-like
        A list of (1, CH, H, W) images

    model_path : str
        The path to a pre-trained model

    Returns
    -------
    masks : list
        A list of (H, W) predicted masks
    """
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)
    xs = [torch.from_numpy(array) for array in image_arrays]
    xs = [x.to(torch.float32).to(device) for x in xs]
    xs = [x - x.min() for x in xs]
    xs = [x / x.max() for x in xs]
    xs = [(x - 0.5) * 2 for x in xs]
    trainer = Trainer.resume_from_path(
        model_path, inference=True, new_exp=None, device=device
    )
    masks = []
    for idx, x in enumerate(xs):
        if verbose > 0:
            print(idx, "/", len(xs), end="\r")
        m = trainer.G.mask(x=x)
        masks.append(m.squeeze().cpu())
    return masks


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    tmp_dir = Path(os.environ["SLURM_TMPDIR"])

    # Build paths to data
    imgs_paths = sorted(find_images(args.images_dir, recursive=False))
    labels_paths = sorted(find_images(args.labels_dir, recursive=False))
    if args.limit > 0:
        imgs_paths = imgs_paths[: args.limit]
        labels_paths = labels_paths[: args.limit]

    print(f"Loaded {len(imgs_paths)} images and labels")

    # Pre-process images: resize + crop
    # TODO: ? make cropping more flexible, not only central
    img_preprocessing = PrepareTest(target_size=args.image_size, normalize=False)
    imgs = img_preprocessing(imgs_paths, normalize=True, rescale=False)
    labels = img_preprocessing(labels_paths, normalize=False, rescale=False)

    # RGBA to RGB
    print("RGBA to RGB", end="", flush=True)
    imgs = [np.squeeze(np.moveaxis(img.numpy(), 1, -1)) for img in imgs]
    imgs = [rgba2rgb(img) if img.shape[-1] == 4 else img for img in imgs]
    imgs = [np.expand_dims(np.moveaxis(img, -1, 0), axis=0) for img in imgs]
    print(" Done.")

    # Encode labels
    print("Encode labels", end="", flush=True)
    labels = [
        encode_mask_label(
            np.squeeze(np.moveaxis(label.numpy().astype(np.uint8), 1, -1)), "flood"
        )
        for label in labels
    ]
    print(" Done.")

    # Obtain mask predictions
    print("Obtain mask predictions", end="", flush=True)
    if not os.path.isdir(args.model):
        preds_paths = sorted(find_images(args.preds_dir, recursive=False))
        preds = img_preprocessing(preds_paths)
        preds = [
            np.squeeze(np.divide(pred.numpy(), np.max(pred.numpy()))[:, 0, :, :])
            for pred in preds
        ]
    else:
        preds = get_inferences(imgs, args.model)
        preds = [pred.numpy() for pred in preds]
    print(" Done.")

    if args.bin_value > 0:
        preds = [pred > args.bin_value for pred in preds]

    # Compute metrics
    df = pd.DataFrame(
        columns=[
            "fpr",
            "fnr",
            "mnr",
            "mpr",
            "tpr",
            "tnr",
            "precision",
            "f1",
            "filename",
        ]
    )

    for idx, (img, label, pred) in enumerate(zip(*(imgs, labels, preds))):
        img = np.squeeze(img)
        label = np.squeeze(label)

        fp_map, fpr = pred_cannot(pred, label, label_cannot=0)
        fn_map, fnr = missed_must(pred, label, label_must=1)
        may_neg_map, may_pos_map, mnr, mpr = may_flood(pred, label, label_may=2)
        tpr, tnr, precision, f1 = masker_metrics(
            pred, label, label_cannot=0, label_must=1
        )

        df.loc[idx] = pd.Series(
            {
                "fpr": fpr,
                "fnr": fnr,
                "mnr": mnr,
                "mpr": mpr,
                "tpr": tpr,
                "tnr": tnr,
                "precision": precision,
                "f1": f1,
                "filename": os.path.basename(imgs_paths[idx]),
            }
        )

    exp = Experiment(project_name="omnigan-masker-metrics")
    exp.log_table("csv", df)
    exp.log_html(df.to_html(col_space="80px"))
    exp.log_metrics(dict(df.mean(0)))
    exp.log_parameters(vars(args))

    plot_paths = plot_labels_images("..", tmp_dir)
    for pp in plot_paths:
        exp.log_image(pp)
