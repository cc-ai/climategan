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
import yaml
import pandas as pd
from skimage.color import rgba2rgb

import matplotlib.pyplot as plt

import torch

from omnigan.data import encode_mask_label
from omnigan.utils import find_images
from omnigan.trainer import Trainer
from omnigan.transforms import PrepareTest
from omnigan.eval_metrics import (
    pred_cannot,
    missed_must,
    may_flood,
    masker_metrics,
    get_confusion_matrix,
    edges_coherence_std_min,
)

print("Ok.")


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Path to a pre-trained model",
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
        "--max_files", default=-1, type=int, help="Limit loaded samples",
    )
    parser.add_argument(
        "--bin_value", default=-1, type=float, help="Mask binarization threshold"
    )
    parser.add_argument(
        "-y",
        "--yaml",
        default=None,
        type=str,
        help="load a yaml file to parametrize the evaluation",
    )

    return parser.parse_args()


def plot_images(
    output_filename,
    img,
    label,
    pred,
    fp_map,
    fn_map,
    may_neg_map,
    may_pos_map,
    edge_coherence=-1,
    pred_edge=None,
    label_edge=None,
    dpi=300,
    alpha=0.5,
    vmin=0.0,
    vmax=1.0,
    fontsize="xx-small",
    cmap={
        "fp": "Reds",
        "fn": "Reds",
        "may_neg": "Oranges",
        "may_pos": "Purples",
        "pred": "Greens",
    },
):
    f, axes = plt.subplots(1, 5, dpi=dpi)

    # FPR (predicted mask on cannot flood)
    axes[0].imshow(img)
    fp_map_plt = axes[0].imshow(
        fp_map, vmin=vmin, vmax=vmax, cmap=cmap["fp"], alpha=alpha
    )
    axes[0].axis("off")
    axes[0].set_title("FPR: {:.4f}".format(fpr), fontsize=fontsize)

    # FNR (missed mask on must flood)
    axes[1].imshow(img)
    fn_map_plt = axes[1].imshow(
        fn_map, vmin=vmin, vmax=vmax, cmap=cmap["fn"], alpha=alpha
    )
    axes[1].axis("off")
    axes[1].set_title("FNR: {:.4f}".format(fnr), fontsize=fontsize)

    # May flood
    axes[2].imshow(img)
    if edge_coherence != -1:
        title = "MNR: {:.2f} | MPR: {:.2f}\nEdge coh.: {:.4f}".format(
            mnr, mpr, edge_coherence
        )
    #         alpha_here = alpha / 4.
    #         pred_edge_plt = axes[2].imshow(
    #             1.0 - pred_edge, cmap="gray", alpha=alpha_here
    #         )
    #         label_edge_plt = axes[2].imshow(
    #             1.0 - label_edge, cmap="gray", alpha=alpha_here
    #         )
    else:
        title = "MNR: {:.2f} | MPR: {:.2f}".format(mnr, mpr)
    #         alpha_here = alpha / 2.
    may_neg_map_plt = axes[2].imshow(
        may_neg_map, vmin=vmin, vmax=vmax, cmap=cmap["may_neg"], alpha=alpha
    )
    may_pos_map_plt = axes[2].imshow(
        may_pos_map, vmin=vmin, vmax=vmax, cmap=cmap["may_pos"], alpha=alpha
    )
    axes[2].set_title(title, fontsize=fontsize)
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(img)
    pred_mask = axes[3].imshow(
        pred, vmin=vmin, vmax=vmax, cmap=cmap["pred"], alpha=alpha
    )
    axes[3].set_title("Predicted mask", fontsize=fontsize)
    axes[3].axis("off")

    # Labels
    axes[4].imshow(img)
    label_mask = axes[4].imshow(label, alpha=alpha)
    axes[4].set_title("Labels", fontsize=fontsize)
    axes[4].axis("off")

    f.savefig(
        output_filename,
        dpi=f.dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(f)


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

    # Determine output dir
    try:
        tmp_dir = Path(os.environ["SLURM_TMPDIR"])
    except:
        tmp_dir = input("Enter tmp output directory: ")
    plot_dir = tmp_dir.joinpath("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Build paths to data
    imgs_paths = sorted(find_images(args.images_dir, recursive=False))
    labels_paths = sorted(find_images(args.labels_dir, recursive=False))
    if args.max_files > 0:
        imgs_paths = imgs_paths[: args.max_files]
        labels_paths = labels_paths[: args.max_files]

    print(f"Loaded {len(imgs_paths)} images and labels")

    # Pre-process images: resize + crop
    # TODO: ? make cropping more flexible, not only central
    img_preprocessing = PrepareTest(target_size=args.image_size)
    imgs = img_preprocessing(imgs_paths, normalize=False, rescale=False)
    labels = img_preprocessing(labels_paths, normalize=False, rescale=False)

    # RGBA to RGB
    print("RGBA to RGB", end="", flush=True)
    imgs = [
        np.squeeze(np.moveaxis(img.numpy().astype(np.uint8), 1, -1)) for img in imgs
    ]
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

    evaluations = []
    if args.yaml:
        y_path = Path(args.yaml)
        assert y_path.exists()
        assert y_path.suffix in {".yaml", ".yml"}
        with y_path.open("r") as f:
            data = yaml.safe_load(f)
        assert "models" in data or "preds_dirs" in data

        if "models" in data:
            evaluations += [
                {"eval_type": "model", "eval_path": m} for m in data["models"]
            ]
        if "preds_dirs" in data:
            evaluations += [
                {"eval_type": "preds_dir", "eval_path": p} for p in data["preds_dirs"]
            ]
    else:
        if not os.path.isdir(args.model):
            evaluations += [{"eval_type": "preds_dir", "eval_path": args.preds_dir}]
        else:
            evaluations += [{"eval_type": "preds_dir", "eval_path": args.model}]

    # Initialize Comet Experiment

    for e, eval_item in enumerate(evaluations):
        print("\n>>>>> Evaluation", e, ":", eval_item["eval_path"])
        print("=" * 50)
        print("=" * 50)
        exp = Experiment(project_name="omnigan-masker-metrics")

        # Obtain mask predictions
        print("Obtain mask predictions", end="", flush=True)
        if eval_item["eval_type"] == "preds_dir":
            preds_paths = sorted(find_images(eval_item["eval_path"], recursive=False))
            if args.max_files > 0:
                preds_paths = preds_paths[: args.max_files]
            preds = img_preprocessing(preds_paths)
            preds = [
                np.squeeze(np.divide(pred.numpy(), np.max(pred.numpy()))[:, 0, :, :])
                for pred in preds
            ]
        else:
            preds = get_inferences(imgs, eval_item["eval_path"], verbose=1)
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
                "edge_coherence",
                "filename",
            ]
        )

        print("Compute metrics and plot images", end="", flush=True)
        for idx, (img, label, pred) in enumerate(zip(*(imgs, labels, preds))):
            print(idx, "/", len(imgs), end="\r")
            img = np.moveaxis(np.squeeze(img), 0, -1)
            label = np.squeeze(label)

            # Basic metrics
            fp_map, fpr = pred_cannot(pred, label, label_cannot=0)
            fn_map, fnr = missed_must(pred, label, label_must=1)
            may_neg_map, may_pos_map, mnr, mpr = may_flood(pred, label, label_may=2)
            tpr, tnr, precision, f1 = masker_metrics(
                pred, label, label_cannot=0, label_must=1
            )

            # Edges coherence
            edge_coherence, pred_edge, label_edge = edges_coherence_std_min(pred, label)

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
                    "edge_coherence": edge_coherence,
                    "filename": os.path.basename(imgs_paths[idx]),
                }
            )

            # Confusion matrix
            confmat, _ = get_confusion_matrix(tpr, tnr, fpr, fnr, mpr, mnr)
            confmat = np.around(confmat, decimals=3)
            exp.log_confusion_matrix(
                file_name=Path(imgs_paths[idx].name + ".json"),
                title=imgs_paths[idx].name,
                matrix=confmat,
                labels=["Cannot", "Must", "May"],
                row_label="Predicted",
                column_label="Ground truth",
            )

            # Plot prediction images
            fig_filename = plot_dir.joinpath(imgs_paths[idx].name)
            plot_images(
                fig_filename,
                img,
                label,
                pred,
                fp_map,
                fn_map,
                may_neg_map,
                may_pos_map,
                edge_coherence,
                pred_edge,
                label_edge,
            )
            exp.log_image(fig_filename)
        print(" Done.")

        # Summary statistics
        means = df.mean(axis=0)
        confmat_mean, confmat_std = get_confusion_matrix(
            df.tpr, df.tnr, df.fpr, df.fnr, df.mpr, df.mnr
        )
        confmat_mean = np.around(confmat_mean, decimals=3)
        confmat_std = np.around(confmat_std, decimals=3)

        # Log to comet
        exp.log_confusion_matrix(
            file_name="confusion_matrix_mean.json",
            title="confusion_matrix_mean.json",
            matrix=confmat_mean,
            labels=["Cannot", "Must", "May"],
            row_label="Predicted",
            column_label="Ground truth",
        )
        exp.log_confusion_matrix(
            file_name="confusion_matrix_std.json",
            title="confusion_matrix_std.json",
            matrix=confmat_std,
            labels=["Cannot", "Must", "May"],
            row_label="Predicted",
            column_label="Ground truth",
        )
        exp.log_table("metrics.csv", df)
        exp.log_html(df.to_html(col_space="80px"))
        exp.log_metrics(dict(means))
        exp.log_parameters(vars(args))
        exp.log_parameters(eval_item)
        exp.add_tag("eval_masker")
