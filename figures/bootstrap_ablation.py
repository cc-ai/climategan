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
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import trim_mean
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------
# -----  Constants  -----
# -----------------------

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
        "--technique",
        default=None,
        type=str,
        help="Keyword specifying the technique. One of: pseudo, depth, segmentation, dada_seg, dada_masker, spade",
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


def add_ci_mean(
    ax, sample_measure, bs_mean, bs_std, ci, color, alpha, fontsize, invert=False
):

    # Fill area between CI
    dist = ax.lines[0]
    dist_y = dist.get_ydata()
    dist_x = dist.get_xdata()
    linewidth = dist.get_linewidth()

    x_idx_low = np.argmin(np.abs(dist_x - ci[0]))
    x_idx_high = np.argmin(np.abs(dist_x - ci[1]))
    x_ci = dist_x[x_idx_low:x_idx_high]
    y_ci = dist_y[x_idx_low:x_idx_high]

    ax.fill_between(x_ci, 0, y_ci, facecolor=color, alpha=alpha)

    # Add vertical lines of CI
    ax.vlines(
        x=ci[0],
        ymin=0.0,
        ymax=y_ci[0],
        color=color,
        linewidth=linewidth,
        label="ci_low",
    )
    ax.vlines(
        x=ci[1],
        ymin=0.0,
        ymax=y_ci[-1],
        color=color,
        linewidth=linewidth,
        label="ci_high",
    )

    # Add annotations
    bbox_props = dict(boxstyle="round, pad=0.4", fc="w", ec="k", lw=2)

    if invert:
        ha_l = "right"
        ha_u = "left"
    else:
        ha_l = "left"
        ha_u = "right"
    ax.text(
        ci[0],
        0.0,
        s="L = {:.4f}".format(ci[0]),
        ha=ha_l,
        va="bottom",
        fontsize=fontsize,
        bbox=bbox_props,
    )
    ax.text(
        ci[1],
        0.0,
        s="U = {:.4f}".format(ci[1]),
        ha=ha_u,
        va="bottom",
        fontsize=fontsize,
        bbox=bbox_props,
    )

    # Add vertical line of bootstrap mean
    x_idx_mean = np.argmin(np.abs(dist_x - bs_mean))
    ax.vlines(
        x=bs_mean, ymin=0.0, ymax=dist_y[x_idx_mean], color="k", linewidth=linewidth
    )

    # Add annotation of bootstrap mean
    bbox_props = dict(boxstyle="round, pad=0.4", fc="w", ec="k", lw=2)

    ax.text(
        bs_mean,
        0.6 * dist_y[x_idx_mean],
        s="Bootstrap mean = {:.4f}".format(bs_mean),
        ha="center",
        va="center",
        fontsize=fontsize,
        bbox=bbox_props,
    )

    # Add vertical line of sample_measure
    x_idx_smeas = np.argmin(np.abs(dist_x - sample_measure))
    ax.vlines(
        x=sample_measure,
        ymin=0.0,
        ymax=dist_y[x_idx_smeas],
        color="k",
        linewidth=linewidth,
        linestyles="dotted",
    )

    # Add SD
    bbox_props = dict(boxstyle="darrow, pad=0.4", fc="w", ec="k", lw=2)

    ax.text(
        bs_mean,
        0.4 * dist_y[x_idx_mean],
        s="SD = {:.4f} = SE".format(bs_std),
        ha="center",
        va="center",
        fontsize=fontsize,
        bbox=bbox_props,
    )


def add_null_pval(ax, null, color, alpha, fontsize):

    # Fill area between CI
    dist = ax.lines[0]
    dist_y = dist.get_ydata()
    dist_x = dist.get_xdata()
    linewidth = dist.get_linewidth()

    x_idx_null = np.argmin(np.abs(dist_x - null))
    if x_idx_null >= (len(dist_x) / 2.0):
        x_pval = dist_x[x_idx_null:]
        y_pval = dist_y[x_idx_null:]
    else:
        x_pval = dist_x[:x_idx_null]
        y_pval = dist_y[:x_idx_null]

    ax.fill_between(x_pval, 0, y_pval, facecolor=color, alpha=alpha)

    # Add vertical lines of null
    dist = ax.lines[0]
    linewidth = dist.get_linewidth()
    y_max = ax.get_ylim()[1]
    ax.vlines(
        x=null,
        ymin=0.0,
        ymax=y_max,
        color="k",
        linewidth=linewidth,
        linestyles="dotted",
    )

    # Add annotations
    bbox_props = dict(boxstyle="round, pad=0.4", fc="w", ec="k", lw=2)

    ax.text(
        null,
        0.75 * y_max,
        s="Null hypothesis = {:.1f}".format(null),
        ha="center",
        va="center",
        fontsize=fontsize,
        bbox=bbox_props,
    )


def plot_bootstrap_distr(
    sample_measure, bs_samples, alpha, color_ci, color_pval=None, null=None
):

    # Compute results from bootstrap
    q_low = (1.0 - alpha) / 2.0
    q_high = 1.0 - q_low
    ci = np.quantile(bs_samples, [q_low, q_high])
    bs_mean = np.mean(bs_samples)
    bs_std = np.std(bs_samples)

    if null is not None and color_pval is not None:
        pval_flag = True
        pval = np.min([[np.mean(bs_samples > null), np.mean(bs_samples < null)]]) * 2
    else:
        pval_flag = False

    # Set up plot
    sns.set(style="whitegrid")
    fontsize = 24
    font = {"family": "DejaVu Sans", "weight": "normal", "size": fontsize}
    plt.rc("font", **font)
    alpha_plot = 0.5

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(30, 12), dpi=args.dpi)

    # Plot distribution of bootstrap means
    sns.kdeplot(bs_samples, color="b", linewidth=5, gridsize=1000, ax=ax)

    y_lim = ax.get_ylim()

    # Change spines
    sns.despine(left=True, bottom=True)

    # Annotations
    add_ci_mean(
        ax,
        sample_measure,
        bs_mean,
        bs_std,
        ci,
        color=color_ci,
        alpha=alpha_plot,
        fontsize=fontsize,
    )

    if pval_flag:
        add_null_pval(ax, null, color=color_pval, alpha=alpha_plot, fontsize=fontsize)

    # Legend
    ci_patch = mpatches.Patch(
        facecolor=color_ci,
        edgecolor=None,
        alpha=alpha_plot,
        label="{:d} % confidence interval".format(int(100 * alpha)),
    )

    if pval_flag:
        if pval == 0.0:
            pval_patch = mpatches.Patch(
                facecolor=color_pval,
                edgecolor=None,
                alpha=alpha_plot,
                label="P value / 2 = {:.1f}".format(pval / 2.0),
            )
        elif np.around(pval / 2.0, decimals=4) > 0.0000:
            pval_patch = mpatches.Patch(
                facecolor=color_pval,
                edgecolor=None,
                alpha=alpha_plot,
                label="P value / 2 = {:.4f}".format(pval / 2.0),
            )
        else:
            pval_patch = mpatches.Patch(
                facecolor=color_pval,
                edgecolor=None,
                alpha=alpha_plot,
                label="P value / 2 < $10^{}$".format(np.ceil(np.log10(pval / 2.0))),
            )

        leg = ax.legend(
            handles=[ci_patch, pval_patch],
            ncol=1,
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            title="",
            fontsize=fontsize,
            columnspacing=1.0,
            labelspacing=0.2,
            markerfirst=True,
        )
    else:
        leg = ax.legend(
            handles=[ci_patch],
            ncol=1,
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            title="",
            fontsize=fontsize,
            columnspacing=1.0,
            labelspacing=0.2,
            markerfirst=True,
        )

    plt.setp(leg.get_title(), fontsize=fontsize, horizontalalignment="left")

    # Set X-label
    ax.set_xlabel("Bootstrap estimates", rotation=0, fontsize=fontsize, labelpad=10.0)

    # Set Y-label
    ax.set_ylabel("Density", rotation=90, fontsize=fontsize, labelpad=10.0)

    # Ticks
    plt.setp(ax.get_xticklabels(), fontsize=0.8 * fontsize, verticalalignment="top")
    plt.setp(ax.get_yticklabels(), fontsize=0.8 * fontsize)

    ax.set_ylim(y_lim)

    return fig, bs_mean, bs_std, ci, pval


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
    output_yml = output_dir / "{}_bootstrap.yml".format(args.technique)
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Determine technique
    if args.technique.lower() not in dict_techniques:
        raise ValueError("{} is not a valid technique".format(args.technique))
    else:
        technique = dict_techniques[args.technique.lower()]

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col="model_img_idx")

    # Find relevant model pairs
    model_pairs = []
    for mi in df.loc[df[technique]].model_feats.unique():
        for mj in df.model_feats.unique():
            if mj == mi:
                continue

            if df.loc[df.model_feats == mj, technique].unique()[0]:
                continue

            is_pair = True
            for f in model_feats:
                if f == technique:
                    continue
                elif (
                    df.loc[df.model_feats == mj, f].unique()[0]
                    != df.loc[df.model_feats == mi, f].unique()[0]
                ):
                    is_pair = False
                    break
                else:
                    pass
            if is_pair:
                model_pairs.append((mi, mj))
                break

    print("\nModel pairs identified:\n")
    for pair in model_pairs:
        print("{} & {}".format(pair[0], pair[1]))

    df["base"] = ["N/A"] * len(df)
    for spp in model_pairs:
        df.loc[df.model_feats.isin(spp), "depth_base"] = spp[1]

    # Build bootstrap data
    data = {m: [] for m in dict_metrics["key_metrics"]}
    for m_with, m_without in model_pairs:
        df_with = df.loc[df.model_feats == m_with]
        df_without = df.loc[df.model_feats == m_without]
        for metric in data.keys():
            diff = (
                df_with.sort_values(by="img_idx")[metric].values
                - df_without.sort_values(by="img_idx")[metric].values
            )
            data[metric].extend(diff.tolist())

    # Run bootstrap
    measures = ["mean", "median", "20_trimmed_mean"]
    bs_data = {meas: {m: np.zeros(args.n_bs) for m in data.keys()} for meas in measures}

    np.random.seed(args.bs_seed)
    for m, data_m in data.items():
        for idx, s in enumerate(tqdm(range(args.n_bs))):
            # Sample with replacement
            bs_sample = np.random.choice(data_m, size=len(data_m), replace=True)

            # Store mean
            bs_data["mean"][m][idx] = np.mean(bs_sample)

            # Store median
            bs_data["median"][m][idx] = np.median(bs_sample)

            # Store 20 % trimmed mean
            bs_data["20_trimmed_mean"][m][idx] = trim_mean(bs_sample, 0.2)

for metric in dict_metrics["key_metrics"]:
    sample_measure = trim_mean(data[metric], 0.2)
    fig, bs_mean, bs_std, ci, pval = plot_bootstrap_distr(
        sample_measure,
        bs_data["20_trimmed_mean"][metric],
        alpha=args.alpha,
        color_ci=color_cat1_light,
        color_pval=color_cat2_light,
        null=0.0,
    )

    # Save figure
    output_fig = output_dir / "{}_bootstrap_{}_{}.png".format(
        args.technique, metric, "20_trimmed_mean"
    )
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")

    # Store results
    output_results = output_dir / "{}_bootstrap_{}_{}.yml".format(
        args.technique, metric, "20_trimmed_mean"
    )
    results_dict = {
        "measure": "20_trimmed_mean",
        "sample_measure": float(sample_measure),
        "bs_mean": float(bs_mean),
        "bs_std": float(bs_std),
        "ci_left": float(ci[0]),
        "ci_right": float(ci[1]),
        "pval": float(pval),
    }
    with open(output_results, "w") as f:
        yaml.dump(results_dict, f)
