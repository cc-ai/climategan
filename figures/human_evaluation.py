"""
This script plots the result of the human evaluation on Amazon Mechanical Turk, where
human participants chose between an image from ClimateGAN or from a different method.
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


# -----------------------
# -----  Constants  -----
# -----------------------

comparables_dict = {
    "munit_flooded": "MUNIT",
    "cyclegan": "CycleGAN",
    "instagan": "InstaGAN",
    "instagan_copypaste": "Mask-InstaGAN",
    "painted_ground": "Painted ground",
}


# Colors
palette_colorblind = sns.color_palette("colorblind")
color_climategan = palette_colorblind[9]

palette_colorblind = sns.color_palette("colorblind")
color_munit = palette_colorblind[1]
color_cyclegan = palette_colorblind[2]
color_instagan = palette_colorblind[3]
color_maskinstagan = palette_colorblind[6]
color_paintedground = palette_colorblind[8]
palette_comparables = [
    color_munit,
    color_cyclegan,
    color_instagan,
    color_maskinstagan,
    color_paintedground,
]
palette_comparables_light = [
    sns.light_palette(color, n_colors=3)[1] for color in palette_comparables
]


def parsed_args():
    """
    Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="amt_omni-vs-other.csv",
        type=str,
        help="CSV containing the results of the human evaluation, pre-processed",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="DPI for the output images",
    )
    parser.add_argument(
        "--n_bs",
        default=1e6,
        type=int,
        help="Number of bootrstrap samples",
    )
    parser.add_argument(
        "--bs_seed",
        default=17,
        type=int,
        help="Bootstrap random seed, for reproducibility",
    )

    return parser.parse_args()


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
    output_yml = output_dir / "args_human_evaluation.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv)

    # Sort Y labels
    comparables = df.comparable.unique()
    is_climategan_sum = [
        df.loc[df.comparable == c, "climategan"].sum() for c in comparables
    ]
    comparables = comparables[np.argsort(is_climategan_sum)[::-1]]

    # Plot setup
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
    fontsize = "medium"

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(10.5, 3), dpi=args.dpi)

    # Plot the total (right)
    sns.barplot(
        data=df.loc[df.is_valid],
        x="is_valid",
        y="comparable",
        order=comparables,
        orient="h",
        label="comparable",
        palette=palette_comparables_light,
        ci=None,
    )

    # Plot the left
    sns.barplot(
        data=df.loc[df.is_valid],
        x="climategan",
        y="comparable",
        order=comparables,
        orient="h",
        label="climategan",
        color=color_climategan,
        ci=99,
        n_boot=args.n_bs,
        seed=args.bs_seed,
        errcolor="black",
        errwidth=1.5,
        capsize=0.1,
    )

    # Draw line at 0.5
    y = np.arange(ax.get_ylim()[1] + 0.1, ax.get_ylim()[0], 0.1)
    x = 0.5 * np.ones(y.shape[0])
    ax.plot(x, y, linestyle=":", linewidth=1.5, color="black")

    # Change Y-Tick labels
    yticklabels = [comparables_dict[ytick.get_text()] for ytick in ax.get_yticklabels()]
    yticklabels_text = ax.set_yticklabels(
        yticklabels, fontsize=fontsize, horizontalalignment="right", x=0.96
    )
    for ytl in yticklabels_text:
        ax.add_artist(ytl)

    # Remove Y-label
    ax.set_ylabel(ylabel="")

    # Change X-Tick labels
    xlim = [0.0, 1.1]
    xticks = np.arange(xlim[0], xlim[1], 0.1)
    ax.set(xticks=xticks)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    # Set X-label
    ax.set_xlabel(None)

    # Change spines
    sns.despine(left=True, bottom=True)

    # Save figure
    output_fig = output_dir / "human_evaluation_rate_climategan.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
