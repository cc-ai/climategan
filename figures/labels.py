"""
This scripts plots images from the Masker test set overlaid with their labels.
"""
print("Imports...", end="")
from argparse import ArgumentParser
import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys

sys.path.append("../")

from eval_masker import crop_and_resize


# -----------------------
# -----  Constants  -----
# -----------------------

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
        "--masker_test_set_dir",
        default=None,
        type=str,
        help="Directory containing the test images",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="List of image file names to plot",
        default=[],
        type=str,
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

    return parser.parse_args()


def map_color(arr, input_color, output_color, rtol=1e-09):
    """
    Maps one color to another
    """
    input_color_arr = np.tile(input_color, (arr.shape[:2] + (1,)))
    output = arr.copy()
    output[np.all(np.isclose(arr, input_color_arr, rtol=rtol), axis=2)] = output_color
    return output


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

    fig, axes = plt.subplots(
        nrows=1, ncols=len(args.images), dpi=args.dpi, figsize=(len(args.images) * 5, 5)
    )

    for idx, img_filename in enumerate(args.images):

        # Read images
        img_path = imgs_orig_path / img_filename
        label_path = labels_path / "{}_labeled.png".format(Path(img_filename).stem)
        img, label = crop_and_resize(img_path, label_path)

        # Map label colors
        label_colmap = label.astype(float)
        label_colmap = map_color(label_colmap, (255, 0, 0), color_cannot)
        label_colmap = map_color(label_colmap, (0, 0, 255), color_must)
        label_colmap = map_color(label_colmap, (0, 0, 0), color_may)

        ax = axes[idx]
        ax.imshow(img)
        ax.imshow(label_colmap, alpha=args.alpha)
        ax.axis("off")

    # Legend
    handles = []
    lw = 1.0
    handles.append(
        mpatches.Patch(
            facecolor=color_must, label="must", linewidth=lw, alpha=args.alpha
        )
    )
    handles.append(
        mpatches.Patch(facecolor=color_may, label="may", linewidth=lw, alpha=args.alpha)
    )
    handles.append(
        mpatches.Patch(
            facecolor=color_cannot, label="cannot", linewidth=lw, alpha=args.alpha
        )
    )
    labels = ["Must-be-flooded", "May-be-flooded", "Cannot-be-flooded"]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.0, 0.85, 1.0, 0.075),
        ncol=len(args.images),
        fontsize="medium",
        frameon=False,
    )

    # Save figure
    output_fig = output_dir / "labels.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
