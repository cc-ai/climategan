"""
This scripts plots examples of the images that get best and worse metrics
"""
print("Imports...", end="")
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from imageio import imread
from skimage.color import rgba2rgb
from sklearn.metrics.pairwise import euclidean_distances

sys.path.append("../")

from climategan.data import encode_mask_label
from climategan.eval_metrics import edges_coherence_std_min
from eval_masker import crop_and_resize

# -----------------------
# -----  Constants  -----
# -----------------------

# Metrics
metrics = ["error", "f05", "edge_coherence"]

dict_metrics = {
    "names": {
        "tpr": "TPR, Recall, Sensitivity",
        "tnr": "TNR, Specificity, Selectivity",
        "fpr": "FPR",
        "fpt": "False positives relative to image size",
        "fnr": "FNR, Miss rate",
        "fnt": "False negatives relative to image size",
        "mpr": "May positive rate (MPR)",
        "mnr": "May negative rate (MNR)",
        "accuracy": "Accuracy (ignoring may)",
        "error": "Error",
        "f05": "F05 score",
        "precision": "Precision",
        "edge_coherence": "Edge coherence",
        "accuracy_must_may": "Accuracy (ignoring cannot)",
    },
    "key_metrics": ["error", "f05", "edge_coherence"],
}


# Colors
colorblind_palette = sns.color_palette("colorblind")
color_cannot = colorblind_palette[1]
color_must = colorblind_palette[2]
color_may = colorblind_palette[7]
color_pred = colorblind_palette[4]

icefire = sns.color_palette("icefire", as_cmap=False, n_colors=5)
color_tp = icefire[0]
color_tn = icefire[1]
color_fp = icefire[4]
color_fn = icefire[3]


def parsed_args():
    """
    Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="ablations_metrics_20210311.csv",
        type=str,
        help="CSV containing the results of the ablation study",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--models_log_path",
        default=None,
        type=str,
        help="Path containing the log files of the models",
    )
    parser.add_argument(
        "--masker_test_set_dir",
        default=None,
        type=str,
        help="Directory containing the test images",
    )
    parser.add_argument(
        "--best_model",
        default="dada, msd_spade, pseudo",
        type=str,
        help="The string identifier of the best model",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="DPI for the output images",
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="Transparency of labels shade",
    )
    parser.add_argument(
        "--percentile",
        default=0.05,
        type=float,
        help="Transparency of labels shade",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Bootstrap random seed, for reproducibility",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        default=False,
        help="Do not generate images",
    )

    return parser.parse_args()


def map_color(arr, input_color, output_color, rtol=1e-09):
    """
    Maps one color to another
    """
    input_color_arr = np.tile(input_color, (arr.shape[:2] + (1,)))
    output = arr.copy()
    output[np.all(np.isclose(arr, input_color_arr, rtol=rtol), axis=2)] = output_color
    return output


def plot_labels(ax, img, label, img_id, do_legend):
    label_colmap = label.astype(float)
    label_colmap = map_color(label_colmap, (255, 0, 0), color_cannot)
    label_colmap = map_color(label_colmap, (0, 0, 255), color_must)
    label_colmap = map_color(label_colmap, (0, 0, 0), color_may)

    ax.imshow(img)
    ax.imshow(label_colmap, alpha=0.5)
    ax.axis("off")

    # Annotation
    ax.annotate(
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        text=img_id,
        fontsize="x-large",
        verticalalignment="top",
        color="white",
    )

    # Legend
    if do_legend:
        handles = []
        lw = 1.0
        handles.append(
            mpatches.Patch(facecolor=color_must, label="must", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(facecolor=color_may, label="must", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(
                facecolor=color_cannot, label="must", linewidth=lw, alpha=0.66
            )
        )
        labels = ["Must-be-flooded", "May-be-flooded", "Cannot-be-flooded"]
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.075),
            ncol=3,
            mode="expand",
            fontsize="xx-small",
            frameon=False,
        )


def plot_pred(ax, img, pred, img_id, do_legend):
    pred = np.tile(np.expand_dims(pred, axis=2), reps=(1, 1, 3))

    pred_colmap = pred.astype(float)
    pred_colmap = map_color(pred_colmap, (1, 1, 1), color_pred)
    pred_colmap_ma = np.ma.masked_not_equal(pred_colmap, color_pred)
    pred_colmap_ma = pred_colmap_ma.mask * img + pred_colmap_ma

    ax.imshow(img)
    ax.imshow(pred_colmap_ma, alpha=0.5)
    ax.axis("off")

    # Annotation
    ax.annotate(
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        text=img_id,
        fontsize="x-large",
        verticalalignment="top",
        color="white",
    )

    # Legend
    if do_legend:
        handles = []
        lw = 1.0
        handles.append(
            mpatches.Patch(facecolor=color_pred, label="must", linewidth=lw, alpha=0.66)
        )
        labels = ["Prediction"]
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.075),
            ncol=3,
            mode="expand",
            fontsize="xx-small",
            frameon=False,
        )


def plot_correct_incorrect(ax, img_filename, img, label, img_id, do_legend):
    # FP
    fp_map = imread(
        model_path / "eval-metrics/fp" / "{}_fp.png".format(Path(img_filename).stem)
    )
    fp_map = np.tile(np.expand_dims(fp_map, axis=2), reps=(1, 1, 3))

    fp_map_colmap = fp_map.astype(float)
    fp_map_colmap = map_color(fp_map_colmap, (1, 1, 1), color_fp)

    # FN
    fn_map = imread(
        model_path / "eval-metrics/fn" / "{}_fn.png".format(Path(img_filename).stem)
    )
    fn_map = np.tile(np.expand_dims(fn_map, axis=2), reps=(1, 1, 3))

    fn_map_colmap = fn_map.astype(float)
    fn_map_colmap = map_color(fn_map_colmap, (1, 1, 1), color_fn)

    # TP
    tp_map = imread(
        model_path / "eval-metrics/tp" / "{}_tp.png".format(Path(img_filename).stem)
    )
    tp_map = np.tile(np.expand_dims(tp_map, axis=2), reps=(1, 1, 3))

    tp_map_colmap = tp_map.astype(float)
    tp_map_colmap = map_color(tp_map_colmap, (1, 1, 1), color_tp)

    # TN
    tn_map = imread(
        model_path / "eval-metrics/tn" / "{}_tn.png".format(Path(img_filename).stem)
    )
    tn_map = np.tile(np.expand_dims(tn_map, axis=2), reps=(1, 1, 3))

    tn_map_colmap = tn_map.astype(float)
    tn_map_colmap = map_color(tn_map_colmap, (1, 1, 1), color_tn)

    label_colmap = label.astype(float)
    label_colmap = map_color(label_colmap, (0, 0, 0), color_may)
    label_colmap_ma = np.ma.masked_not_equal(label_colmap, color_may)
    label_colmap_ma = label_colmap_ma.mask * img + label_colmap_ma

    # Combine masks
    maps = fp_map_colmap + fn_map_colmap + tp_map_colmap + tn_map_colmap
    maps_ma = np.ma.masked_equal(maps, (0, 0, 0))
    maps_ma = maps_ma.mask * img + maps_ma

    ax.imshow(img)
    ax.imshow(label_colmap_ma, alpha=0.5)
    ax.imshow(maps_ma, alpha=0.5)
    ax.axis("off")

    # Annotation
    ax.annotate(
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        text=img_id,
        fontsize="x-large",
        verticalalignment="top",
        color="white",
    )

    # Legend
    if do_legend:
        handles = []
        lw = 1.0
        handles.append(
            mpatches.Patch(facecolor=color_tp, label="TP", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(facecolor=color_tn, label="TN", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(facecolor=color_fp, label="FP", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(facecolor=color_fn, label="FN", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(
                facecolor=color_may, label="May-be-flooded", linewidth=lw, alpha=0.66
            )
        )
        labels = ["TP", "TN", "FP", "FN", "May-be-flooded"]
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.075),
            ncol=5,
            mode="expand",
            fontsize="xx-small",
            frameon=False,
        )


def plot_edge_coherence(ax, img, label, pred, img_id, do_legend):
    pred = np.tile(np.expand_dims(pred, axis=2), reps=(1, 1, 3))

    ec, pred_ec, label_ec = edges_coherence_std_min(
        np.squeeze(pred[:, :, 0]), np.squeeze(encode_mask_label(label, "flood"))
    )

    ##################
    # Edge distances #
    ##################

    # Location of edges
    pred_ec_coord = np.argwhere(pred_ec > 0)
    label_ec_coord = np.argwhere(label_ec > 0)

    # Normalized pairwise distances between pred and label
    dist_mat = np.divide(
        euclidean_distances(pred_ec_coord, label_ec_coord), pred_ec.shape[0]
    )

    # Standard deviation of the minimum distance from pred to label
    min_dist = np.min(dist_mat, axis=1)  # noqa: F841

    #############
    # Make plot #
    #############

    pred_ec = np.tile(
        np.expand_dims(np.asarray(pred_ec > 0, dtype=float), axis=2), reps=(1, 1, 3)
    )
    pred_ec_colmap = map_color(pred_ec, (1, 1, 1), color_pred)
    pred_ec_colmap_ma = np.ma.masked_not_equal(pred_ec_colmap, color_pred)  # noqa: F841

    label_ec = np.tile(
        np.expand_dims(np.asarray(label_ec > 0, dtype=float), axis=2), reps=(1, 1, 3)
    )
    label_ec_colmap = map_color(label_ec, (1, 1, 1), color_must)
    label_ec_colmap_ma = np.ma.masked_not_equal(  # noqa: F841
        label_ec_colmap, color_must
    )

    # Combined pred and label edges
    combined_ec = pred_ec_colmap + label_ec_colmap
    combined_ec_ma = np.ma.masked_equal(combined_ec, (0, 0, 0))
    combined_ec_img = combined_ec_ma.mask * img + combined_ec

    # Pred
    pred_colmap = pred.astype(float)
    pred_colmap = map_color(pred_colmap, (1, 1, 1), color_pred)
    pred_colmap_ma = np.ma.masked_not_equal(pred_colmap, color_pred)

    # Must
    label_colmap = label.astype(float)
    label_colmap = map_color(label_colmap, (0, 0, 255), color_must)
    label_colmap_ma = np.ma.masked_not_equal(label_colmap, color_must)

    # TP
    tp_map = imread(
        model_path / "eval-metrics/tp" / "{}_tp.png".format(Path(srs_sel.filename).stem)
    )
    tp_map = np.tile(np.expand_dims(tp_map, axis=2), reps=(1, 1, 3))
    tp_map_colmap = tp_map.astype(float)
    tp_map_colmap = map_color(tp_map_colmap, (1, 1, 1), color_tp)
    tp_map_colmap_ma = np.ma.masked_not_equal(tp_map_colmap, color_tp)

    # Combination
    comb_pred = (
        (pred_colmap_ma.mask ^ tp_map_colmap_ma.mask)
        & tp_map_colmap_ma.mask
        & combined_ec_ma.mask
    ) * pred_colmap
    comb_label = (
        (label_colmap_ma.mask ^ pred_colmap_ma.mask)
        & pred_colmap_ma.mask
        & combined_ec_ma.mask
    ) * label_colmap
    comb_tp = combined_ec_ma.mask * tp_map_colmap.copy()
    combined = comb_tp + comb_label + comb_pred
    combined_ma = np.ma.masked_equal(combined, (0, 0, 0))
    combined_ma = combined_ma.mask * combined_ec_img + combined_ma

    ax.imshow(combined_ec_img, alpha=1)
    ax.imshow(combined_ma, alpha=0.5)
    ax.axis("off")

    # Plot lines
    idx_sort_x = np.argsort(pred_ec_coord[:, 1])
    offset = 100
    for idx in range(offset, pred_ec_coord.shape[0], offset):
        y0, x0 = pred_ec_coord[idx_sort_x[idx], :]
        argmin = np.argmin(dist_mat[idx_sort_x[idx]])
        y1, x1 = label_ec_coord[argmin, :]
        ax.plot([x0, x1], [y0, y1], color="white", linewidth=0.5)

    # Annotation
    ax.annotate(
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        text=img_id,
        fontsize="x-large",
        verticalalignment="top",
        color="white",
    )
    # Legend
    if do_legend:
        handles = []
        lw = 1.0
        handles.append(
            mpatches.Patch(facecolor=color_tp, label="TP", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(facecolor=color_pred, label="pred", linewidth=lw, alpha=0.66)
        )
        handles.append(
            mpatches.Patch(
                facecolor=color_must, label="Must-be-flooded", linewidth=lw, alpha=0.66
            )
        )
        labels = ["TP", "Prediction", "Must-be-flooded"]
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.075),
            ncol=3,
            mode="expand",
            fontsize="xx-small",
            frameon=False,
        )


def plot_images_metric(axes, metric, img_filename, img_id, do_legend):

    # Read images
    img_path = imgs_orig_path / img_filename
    label_path = labels_path / "{}_labeled.png".format(Path(img_filename).stem)
    img, label = crop_and_resize(img_path, label_path)
    img = rgba2rgb(img) if img.shape[-1] == 4 else img / 255.0
    pred = imread(
        model_path / "eval-metrics/pred" / "{}_pred.png".format(Path(img_filename).stem)
    )

    # Label
    plot_labels(axes[0], img, label, img_id, do_legend)

    # Prediction
    plot_pred(axes[1], img, pred, img_id, do_legend)

    # Correct / incorrect
    if metric in ["error", "f05"]:
        plot_correct_incorrect(axes[2], img_filename, img, label, img_id, do_legend)
    # Edge coherence
    elif metric == "edge_coherence":
        plot_edge_coherence(axes[2], img, label, pred, img_id, do_legend)
    else:
        raise ValueError


def scatterplot_metrics_pair(ax, df, x_metric, y_metric, dict_images):

    sns.scatterplot(data=df, x=x_metric, y=y_metric, ax=ax)

    # Set X-label
    ax.set_xlabel(dict_metrics["names"][x_metric], rotation=0, fontsize="medium")

    # Set Y-label
    ax.set_ylabel(dict_metrics["names"][y_metric], rotation=90, fontsize="medium")

    # Change spines
    sns.despine(ax=ax, left=True, bottom=True)

    annotate_scatterplot(ax, dict_images, x_metric, y_metric)


def scatterplot_metrics(ax, df, dict_images):

    sns.scatterplot(data=df, x="error", y="f05", hue="edge_coherence", ax=ax)

    # Set X-label
    ax.set_xlabel(dict_metrics["names"]["error"], rotation=0, fontsize="medium")

    # Set Y-label
    ax.set_ylabel(dict_metrics["names"]["f05"], rotation=90, fontsize="medium")

    annotate_scatterplot(ax, dict_images, "error", "f05")

    # Change spines
    sns.despine(ax=ax, left=True, bottom=True)

    # Set XY limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([0.0, xlim[1]])
    ax.set_ylim([ylim[0], 1.0])


def annotate_scatterplot(ax, dict_images, x_metric, y_metric, offset=0.1):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_len = xlim[1] - xlim[0]
    y_len = ylim[1] - ylim[0]
    x_th = xlim[1] - x_len / 2.0
    y_th = ylim[1] - y_len / 2.0
    for text, d in dict_images.items():
        x = d[x_metric]
        y = d[y_metric]
        x_text = x + x_len * offset if x < x_th else x - x_len * offset
        y_text = y + y_len * offset if y < y_th else y - y_len * offset
        ax.annotate(
            xy=(x, y),
            xycoords="data",
            xytext=(x_text, y_text),
            textcoords="data",
            text=text,
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize="medium",
            color="black",
        )


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------
    args = parsed_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    # Determine output dir
    if args.output_dir is None:
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)

    # Store args
    output_yml = output_dir / "labels.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Data dirs
    imgs_orig_path = Path(args.masker_test_set_dir) / "imgs"
    labels_path = Path(args.masker_test_set_dir) / "labels"

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col="model_img_idx")

    # Select best model
    df = df.loc[df.model_feats == args.best_model]
    v_key, model_dir = df.model.unique()[0].split("/")
    model_path = Path(args.models_log_path) / "ablation-{}".format(v_key) / model_dir

    # Set up plot
    sns.reset_orig()
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "Utopia",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "ITC Bookman",
                "Bookman",
                "Times",
                "Palatino",
                "Charter",
                "serif" "Bitstream Vera Serif",
                "DejaVu Serif",
            ]
        }
    )

    if args.seed:
        np.random.seed(args.seed)
    img_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dict_images = {}
    idx = 0
    for metric in metrics:

        fig, axes = plt.subplots(nrows=2, ncols=3, dpi=200, figsize=(18, 12))

        # Select best
        if metric == "error":
            ascending = True
        else:
            ascending = False
        idx_rand = np.random.permutation(int(args.percentile * len(df)))[0]
        srs_sel = df.sort_values(by=metric, ascending=ascending).iloc[idx_rand]
        img_id = img_ids[idx]
        dict_images.update({img_id: srs_sel})

        # Read images
        img_filename = srs_sel.filename

        if not args.no_images:
            axes_row = axes[0, :]
            plot_images_metric(axes_row, metric, img_filename, img_id, do_legend=True)

        idx += 1

        # Select worst
        if metric == "error":
            ascending = False
        else:
            ascending = True
        idx_rand = np.random.permutation(int(args.percentile * len(df)))[0]
        srs_sel = df.sort_values(by=metric, ascending=ascending).iloc[idx_rand]
        img_id = img_ids[idx]
        dict_images.update({img_id: srs_sel})

        # Read images
        img_filename = srs_sel.filename

        if not args.no_images:
            axes_row = axes[1, :]
            plot_images_metric(axes_row, metric, img_filename, img_id, do_legend=False)

        idx += 1

        # Save figure
        output_fig = output_dir / "{}.png".format(metric)
        fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")

    fig = plt.figure(dpi=200)
    scatterplot_metrics(fig.gca(), df, dict_images)

    #     fig, axes = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(18, 5))
    #
    #     scatterplot_metrics_pair(axes[0], df, 'error', 'f05', dict_images)
    #     scatterplot_metrics_pair(axes[1], df, 'error', 'edge_coherence', dict_images)
    #     scatterplot_metrics_pair(axes[2], df, 'f05', 'edge_coherence', dict_images)
    #
    output_fig = output_dir / "scatterplots.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
