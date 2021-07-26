"""
This script computes the median difference and confidence intervals of all techniques from the ablation study for
improving the masker evaluation metrics. The differences in the metrics are computed
for all images of paired models, that is those which only differ in the inclusion or
not of the given technique. Then, statistical inference is performed through the
percentile bootstrap to obtain robust estimates of the differences in the metrics and
confidence intervals. The script plots the summary for all techniques.
"""
print("Imports...", end="")
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import comb
from scipy.stats import trim_mean
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms


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
    "key_metrics": ["error", "f05", "edge_coherence"],
}

dict_techniques = OrderedDict(
    [
        ("pseudo", "Pseudo labels"),
        ("depth", "Depth (D)"),
        ("seg", "Seg. (S)"),
        ("spade", "SPADE"),
        ("dada_seg", "DADA (S)"),
        ("dada_masker", "DADA (M)"),
    ]
)

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
crest = sns.color_palette("crest", as_cmap=False, n_colors=7)
palette_metrics = [crest[0], crest[3], crest[6]]
sns.palplot(palette_metrics)

# Markers
dict_markers = {"error": "o", "f05": "s", "edge_coherence": "^"}


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


def trim_mean_wrapper(a):
    return trim_mean(a, proportiontocut=0.2)


def find_model_pairs(technique, model_feats):
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
    return model_pairs


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
    output_yml = output_dir / "bootstrap_summary.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col="model_img_idx")

    # Build data set
    dfbs = pd.DataFrame(columns=["diff", "technique", "metric"])
    for technique in model_feats:

        # Get pairs
        model_pairs = find_model_pairs(technique, model_feats)

        # Compute differences
        for m_with, m_without in model_pairs:
            df_with = df.loc[df.model_feats == m_with]
            df_without = df.loc[df.model_feats == m_without]
            for metric in dict_metrics["key_metrics"]:
                diff = (
                    df_with.sort_values(by="img_idx")[metric].values
                    - df_without.sort_values(by="img_idx")[metric].values
                )
                dfm = pd.DataFrame.from_dict(
                    {"metric": metric, "technique": technique, "diff": diff}
                )
                dfbs = dfbs.append(dfm, ignore_index=True)

    ### Plot

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
        nrows=1, ncols=3, sharey=True, dpi=args.dpi, figsize=(9, 3)
    )

    metrics = ["error", "f05", "edge_coherence"]
    dict_ci = {m: {} for m in metrics}

    for idx, metric in enumerate(dict_metrics["key_metrics"]):

        ax = sns.pointplot(
            ax=axes[idx],
            data=dfbs.loc[dfbs.metric.isin(["error", "f05", "edge_coherence"])],
            order=dict_techniques.keys(),
            x="diff",
            y="technique",
            hue="metric",
            hue_order=[metric],
            markers=dict_markers[metric],
            palette=[palette_metrics[idx]],
            errwidth=1.5,
            scale=0.6,
            join=False,
            estimator=trim_mean_wrapper,
            ci=int(args.alpha * 100),
            n_boot=args.n_bs,
            seed=args.bs_seed,
        )

        # Retrieve confidence intervals and update results dictionary
        for line, technique in zip(ax.lines, dict_techniques.keys()):
            dict_ci[metric].update(
                {
                    technique: {
                        "20_trimmed_mean": float(
                            trim_mean_wrapper(
                                dfbs.loc[
                                    (dfbs.technique == technique)
                                    & (dfbs.metric == metrics[idx]),
                                    "diff",
                                ].values
                            )
                        ),
                        "ci_left": float(line.get_xdata()[0]),
                        "ci_right": float(line.get_xdata()[1]),
                    }
                }
            )

        leg_handles, leg_labels = ax.get_legend_handles_labels()

        # Change spines
        sns.despine(left=True, bottom=True)

        # Set Y-label
        ax.set_ylabel(None)

        # Y-tick labels
        ax.set_yticklabels(list(dict_techniques.values()), fontsize="medium")

        # Set X-label
        ax.set_xlabel(None)

        # X-ticks
        xticks = ax.get_xticks()
        xticklabels = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize="small")

        # Y-lim
        display2data = ax.transData.inverted()
        ax2display = ax.transAxes
        _, y_bottom = display2data.transform(ax.transAxes.transform((0.0, 0.02)))
        _, y_top = display2data.transform(ax.transAxes.transform((0.0, 0.98)))
        ax.set_ylim(bottom=y_bottom, top=y_top)

        # Draw line at H0
        y = np.arange(ax.get_ylim()[1], ax.get_ylim()[0], 0.1)
        x = 0.0 * np.ones(y.shape[0])
        ax.plot(x, y, linestyle=":", linewidth=1.5, color="black")

        # Draw gray area
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if metric == "error":
            x0 = xlim[0]
            width = np.abs(x0)
        else:
            x0 = 0.0
            width = np.abs(xlim[1])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        rect = mpatches.Rectangle(
            xy=(x0, 0.0),
            width=width,
            height=1,
            transform=trans,
            linewidth=0.0,
            edgecolor="none",
            facecolor="gray",
            alpha=0.05,
        )
        ax.add_patch(rect)

        # Legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()
        leg_labels = [dict_metrics["names"][metric] for metric in leg_labels]
        leg = ax.legend(
            handles=leg_handles,
            labels=leg_labels,
            loc="center",
            title="",
            bbox_to_anchor=(-0.2, 1.05, 1.0, 0.0),
            framealpha=1.0,
            frameon=False,
            handletextpad=-0.2,
        )

    # Set X-label (title)                                                                                                     │
    fig.suptitle(
        "20 % trimmed mean difference and bootstrapped confidence intervals",
        y=0.0,
        fontsize="medium",
    )

    # Save figure
    output_fig = output_dir / "bootstrap_summary.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")

    # Store results
    output_results = output_dir / "bootstrap_summary_results.yml"
    with open(output_results, "w") as f:
        yaml.dump(dict_ci, f)
