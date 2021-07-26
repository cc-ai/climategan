"""
This script evaluates the contribution of a technique from the ablation study for
improving the masker evaluation metrics. The differences in the metrics are computed
for all images of paired models, that is those which only differ in the inclusion or
not of the given technique. Then, statistical inference is performed through the
percentile bootstrap to obtain robust estimates of the differences in the metrics and
confidence intervals. The script plots the distribution of the bootrstraped estimates.
"""
print("Imports...", end="")
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms


# -----------------------
# -----  Constants  -----
# -----------------------

dict_models = {
    "md": 11,
    "dada_ms, msd, pseudo": 9,
    "msd, pseudo": 4,
    "dada, msd_spade, pseudo": 7,
    "msd": 13,
    "dada_m, msd": 17,
    "dada, msd_spade": 16,
    "msd_spade, pseudo": 5,
    "dada_ms, msd": 18,
    "dada, msd, pseudo": 6,
    "ms": 12,
    "dada, msd": 15,
    "dada_m, msd, pseudo": 8,
    "msd_spade": 14,
    "m": 10,
    "md, pseudo": 2,
    "ms, pseudo": 3,
    "m, pseudo": 1,
    "ground": "G",
    "instagan": "I",
}

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
    "key_metrics": ["f05", "error", "edge_coherence"],
}
dict_techniques = {
    "depth": "depth",
    "segmentation": "seg",
    "seg": "seg",
    "dada_s": "dada_seg",
    "dada_seg": "dada_seg",
    "dada_segmentation": "dada_seg",
    "dada_m": "dada_masker",
    "dada_masker": "dada_masker",
    "spade": "spade",
    "pseudo": "pseudo",
    "pseudo-labels": "pseudo",
    "pseudo_labels": "pseudo",
}

# Markers
dict_markers = {"error": "o", "f05": "s", "edge_coherence": "^"}

# Model features
model_feats = [
    "masker",
    "seg",
    "depth",
    "dada_seg",
    "dada_masker",
    "spade",
    "pseudo",
    "ground",
    "instagan",
]

# Colors
palette_colorblind = sns.color_palette("colorblind")
color_climategan = palette_colorblind[0]
color_munit = palette_colorblind[1]
color_cyclegan = palette_colorblind[6]
color_instagan = palette_colorblind[8]
color_maskinstagan = palette_colorblind[2]
color_paintedground = palette_colorblind[3]

color_cat1 = palette_colorblind[0]
color_cat2 = palette_colorblind[1]
palette_lightest = [
    sns.light_palette(color_cat1, n_colors=20)[3],
    sns.light_palette(color_cat2, n_colors=20)[3],
]
palette_light = [
    sns.light_palette(color_cat1, n_colors=3)[1],
    sns.light_palette(color_cat2, n_colors=3)[1],
]
palette_medium = [color_cat1, color_cat2]
palette_dark = [
    sns.dark_palette(color_cat1, n_colors=3)[1],
    sns.dark_palette(color_cat2, n_colors=3)[1],
]
palette_cat1 = [
    palette_lightest[0],
    palette_light[0],
    palette_medium[0],
    palette_dark[0],
]
palette_cat2 = [
    palette_lightest[1],
    palette_light[1],
    palette_medium[1],
    palette_dark[1],
]
color_cat1_light = palette_light[0]
color_cat2_light = palette_light[1]


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
        "--models",
        default="all",
        type=str,
        help="Models to display: all, pseudo, no_dada_masker, no_baseline",
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
        "--alpha",
        default=0.99,
        type=float,
        help="Confidence level",
    )
    parser.add_argument(
        "--bs_seed",
        default=17,
        type=int,
        help="Bootstrap random seed, for reproducibility",
    )

    return parser.parse_args()


def plot_median_metrics(
    df, do_stripplot=True, dpi=200, bs_seed=37, n_bs=1000, **snskwargs
):
    def plot_metric(
        ax, df, metric, do_stripplot=True, dpi=200, bs_seed=37, marker="o", **snskwargs
    ):

        y_labels = [dict_models[f] for f in df.model_feats.unique()]

        # Labels
        y_labels_int = np.sort([el for el in y_labels if isinstance(el, int)]).tolist()
        y_order_int = [
            k for vs in y_labels_int for k, vu in dict_models.items() if vs == vu
        ]
        y_labels_int = [str(el) for el in y_labels_int]

        y_labels_str = sorted([el for el in y_labels if not isinstance(el, int)])
        y_order_str = [
            k for vs in y_labels_str for k, vu in dict_models.items() if vs == vu
        ]
        y_labels = y_labels_int + y_labels_str
        y_order = y_order_int + y_order_str

        # Palette
        palette = len(y_labels_int) * [color_climategan]
        for y in y_labels_str:
            if y == "G":
                palette = palette + [color_paintedground]
            if y == "I":
                palette = palette + [color_maskinstagan]

        # Error
        sns.pointplot(
            ax=ax,
            data=df,
            x=metric,
            y="model_feats",
            order=y_order,
            markers=marker,
            estimator=np.median,
            ci=99,
            seed=bs_seed,
            n_boot=n_bs,
            join=False,
            scale=0.6,
            errwidth=1.5,
            capsize=0.1,
            palette=palette,
        )
        xlim = ax.get_xlim()

        if do_stripplot:
            sns.stripplot(
                ax=ax,
                data=df,
                x=metric,
                y="model_feats",
                size=1.5,
                palette=palette,
                alpha=0.2,
            )
        ax.set_xlim(xlim)

        # Set X-label
        ax.set_xlabel(dict_metrics["names"][metric], rotation=0, fontsize="medium")

        # Set Y-label
        ax.set_ylabel(None)

        ax.set_yticklabels(y_labels, fontsize="medium")

        # Change spines
        sns.despine(ax=ax, left=True, bottom=True)

        # Draw gray area on final model
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        rect = mpatches.Rectangle(
            xy=(0.0, 5.5),
            width=1,
            height=1,
            transform=trans,
            linewidth=0.0,
            edgecolor="none",
            facecolor="gray",
            alpha=0.05,
        )
        ax.add_patch(rect)

    # Set up plot
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

    fig_h = 0.4 * len(df.model_feats.unique())
    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharey=True, dpi=dpi, figsize=(18, fig_h)
    )

    # Error
    plot_metric(
        axes[0],
        df,
        "error",
        do_stripplot=do_stripplot,
        dpi=dpi,
        bs_seed=bs_seed,
        marker=dict_markers["error"],
    )
    axes[0].set_ylabel("Models")

    # F05
    plot_metric(
        axes[1],
        df,
        "f05",
        do_stripplot=do_stripplot,
        dpi=dpi,
        bs_seed=bs_seed,
        marker=dict_markers["f05"],
    )

    # Edge coherence
    plot_metric(
        axes[2],
        df,
        "edge_coherence",
        do_stripplot=do_stripplot,
        dpi=dpi,
        bs_seed=bs_seed,
        marker=dict_markers["edge_coherence"],
    )
    xticks = axes[2].get_xticks()
    xticklabels = ["{:.3f}".format(x) for x in xticks]
    axes[2].set(xticks=xticks, xticklabels=xticklabels)

    plt.subplots_adjust(wspace=0.12)

    return fig


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
    output_yml = output_dir / "ablation_comparison_{}.yml".format(args.models)
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col="model_img_idx")

    # Determine models
    if "all" in args.models.lower():
        pass
    else:
        if "no_baseline" in args.models.lower():
            df = df.loc[(df.ground == False) & (df.instagan == False)]
        if "pseudo" in args.models.lower():
            df = df.loc[
                (df.pseudo == True) | (df.ground == True) | (df.instagan == True)
            ]
        if "no_dada_mask" in args.models.lower():
            df = df.loc[
                (df.dada_masker == False) | (df.ground == True) | (df.instagan == True)
            ]

    fig = plot_median_metrics(df, do_stripplot=True, dpi=args.dpi, bs_seed=args.bs_seed)

    # Save figure
    output_fig = output_dir / "ablation_comparison_{}.png".format(args.models)
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
