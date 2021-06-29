"""
Compute metrics of the performance of the masker using a set of ground-truth labels

run eval_masker.py --model "/miniscratch/_groups/ccai/checkpoints/model/"

"""
print("Imports...", end="")
import os
import os.path
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comet_ml import Experiment
import torch
import yaml
from skimage.color import rgba2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torchvision.transforms import ToTensor

from climategan.data import encode_mask_label
from climategan.eval_metrics import (
    masker_classification_metrics,
    get_confusion_matrix,
    edges_coherence_std_min,
    boxplot_metric,
    clustermap_metric,
)
from climategan.transforms import PrepareTest
from climategan.trainer import Trainer
from climategan.utils import find_images

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
        "error": "Error (ignoring may)",
        "f05": "F0.05 score",
        "precision": "Precision",
        "edge_coherence": "Edge coherence",
        "accuracy_must_may": "Accuracy (ignoring cannot)",
    },
    "threshold": {
        "tpr": 0.95,
        "tnr": 0.95,
        "fpr": 0.05,
        "fpt": 0.01,
        "fnr": 0.05,
        "fnt": 0.01,
        "accuracy": 0.95,
        "error": 0.05,
        "f05": 0.95,
        "precision": 0.95,
        "edge_coherence": 0.02,
        "accuracy_must_may": 0.5,
    },
    "key_metrics": ["f05", "error", "edge_coherence", "mnr"],
}

print("Ok.")


def parsed_args():
    """Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a pre-trained model",
    )
    parser.add_argument(
        "--images_dir",
        default="/miniscratch/_groups/ccai/data/omnigan/masker-test-set/imgs",
        type=str,
        help="Directory containing the original test images",
    )
    parser.add_argument(
        "--labels_dir",
        default="/miniscratch/_groups/ccai/data/omnigan/masker-test-set/labels",
        type=str,
        help="Directory containing the labeled images",
    )
    parser.add_argument(
        "--image_size",
        default=640,
        type=int,
        help="The height and weight of the pre-processed images",
    )
    parser.add_argument(
        "--max_files",
        default=-1,
        type=int,
        help="Limit loaded samples",
    )
    parser.add_argument(
        "--bin_value", default=0.5, type=float, help="Mask binarization threshold"
    )
    parser.add_argument(
        "-y",
        "--yaml",
        default=None,
        type=str,
        help="load a yaml file to parametrize the evaluation",
    )
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=False,
        help="Plot masker images & their metrics overlays",
    )
    parser.add_argument(
        "--no_paint",
        action="store_true",
        default=False,
        help="Do not log painted images",
    )
    parser.add_argument(
        "--write_metrics",
        action="store_true",
        default=False,
        help="If True, write CSV file and maps images in model's path directory",
    )
    parser.add_argument(
        "--load_metrics",
        action="store_true",
        default=False,
        help="If True, load predictions and metrics instead of re-computing",
    )
    parser.add_argument(
        "--prepare_torch",
        action="store_true",
        default=False,
        help="If True, pre-process images as torch tensors",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=str,
        help="Filename of the output CSV with the metrics of all models",
    )

    return parser.parse_args()


def uint8(array):
    return array.astype(np.uint8)


def crop_and_resize(image_path, label_path):
    """
    Resizes an image so that it keeps the aspect ratio and the smallest dimensions
    is 640, then crops this resized image in its center so that the output is 640x640
    without aspect ratio distortion

    Args:
        image_path (Path or str): Path to an image
        label_path (Path or str): Path to the image's associated label

    Returns:
        tuple((np.ndarray, np.ndarray)): (new image, new label)
    """

    img = imread(image_path)
    lab = imread(label_path)

    # if img.shape[-1] == 4:
    #     img = uint8(rgba2rgb(img) * 255)

    # TODO: remove (debug)
    if img.shape[:2] != lab.shape[:2]:
        print(
            "\nWARNING: shape mismatch: im -> ({}) {}, lab -> ({}) {}".format(
                img.shape[:2], image_path.name, lab.shape[:2], label_path.name
            )
        )
        # breakpoint()

    # resize keeping aspect ratio: smallest dim is 640
    i_h, i_w = img.shape[:2]
    if i_h < i_w:
        i_size = (640, int(640 * i_w / i_h))
    else:
        i_size = (int(640 * i_h / i_w), 640)

    l_h, l_w = img.shape[:2]
    if l_h < l_w:
        l_size = (640, int(640 * l_w / l_h))
    else:
        l_size = (int(640 * l_h / l_w), 640)

    r_img = resize(img, i_size, preserve_range=True, anti_aliasing=True)
    r_img = uint8(r_img)

    r_lab = resize(lab, l_size, preserve_range=True, anti_aliasing=False, order=0)
    r_lab = uint8(r_lab)

    # crop in the center
    H, W = r_img.shape[:2]

    top = (H - 640) // 2
    left = (W - 640) // 2

    rc_img = r_img[top : top + 640, left : left + 640, :]
    rc_lab = (
        r_lab[top : top + 640, left : left + 640, :]
        if r_lab.ndim == 3
        else r_lab[top : top + 640, left : left + 640]
    )

    return rc_img, rc_lab


def plot_images(
    output_filename,
    img,
    label,
    pred,
    metrics_dict,
    maps_dict,
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
    fp_map_plt = axes[0].imshow(  # noqa: F841
        maps_dict["fp"], vmin=vmin, vmax=vmax, cmap=cmap["fp"], alpha=alpha
    )
    axes[0].axis("off")
    axes[0].set_title("FPR: {:.4f}".format(metrics_dict["fpr"]), fontsize=fontsize)

    # FNR (missed mask on must flood)
    axes[1].imshow(img)
    fn_map_plt = axes[1].imshow(  # noqa: F841
        maps_dict["fn"], vmin=vmin, vmax=vmax, cmap=cmap["fn"], alpha=alpha
    )
    axes[1].axis("off")
    axes[1].set_title("FNR: {:.4f}".format(metrics_dict["fnr"]), fontsize=fontsize)

    # May flood
    axes[2].imshow(img)
    if edge_coherence != -1:
        title = "MNR: {:.2f} | MPR: {:.2f}\nEdge coh.: {:.4f}".format(
            metrics_dict["mnr"], metrics_dict["mpr"], edge_coherence
        )
    #         alpha_here = alpha / 4.
    #         pred_edge_plt = axes[2].imshow(
    #             1.0 - pred_edge, cmap="gray", alpha=alpha_here
    #         )
    #         label_edge_plt = axes[2].imshow(
    #             1.0 - label_edge, cmap="gray", alpha=alpha_here
    #         )
    else:
        title = "MNR: {:.2f} | MPR: {:.2f}".format(mnr, mpr)  # noqa: F821
    #         alpha_here = alpha / 2.
    may_neg_map_plt = axes[2].imshow(  # noqa: F841
        maps_dict["may_neg"], vmin=vmin, vmax=vmax, cmap=cmap["may_neg"], alpha=alpha
    )
    may_pos_map_plt = axes[2].imshow(  # noqa: F841
        maps_dict["may_pos"], vmin=vmin, vmax=vmax, cmap=cmap["may_pos"], alpha=alpha
    )
    axes[2].set_title(title, fontsize=fontsize)
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(img)
    pred_mask = axes[3].imshow(  # noqa: F841
        pred, vmin=vmin, vmax=vmax, cmap=cmap["pred"], alpha=alpha
    )
    axes[3].set_title("Predicted mask", fontsize=fontsize)
    axes[3].axis("off")

    # Labels
    axes[4].imshow(img)
    label_mask = axes[4].imshow(label, alpha=alpha)  # noqa: F841
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


def load_ground(ground_output_path, ref_image_path):
    gop = Path(ground_output_path)
    rip = Path(ref_image_path)

    ground_paths = list((gop / "eval-metrics" / "pred").glob(f"{rip.stem}.jpg")) + list(
        (gop / "eval-metrics" / "pred").glob(f"{rip.stem}.png")
    )
    if len(ground_paths) == 0:
        raise ValueError(
            f"Could not find a ground match in {str(gop)} for image {str(rip)}"
        )
    elif len(ground_paths) > 1:
        raise ValueError(
            f"Found more than 1 ground match in {str(gop)} for image {str(rip)}:"
            + f" {list(map(str, ground_paths))}"
        )
    ground_path = ground_paths[0]
    _, ground = crop_and_resize(rip, ground_path)
    if ground.ndim == 3:
        ground = ground[:, :, 0]
    ground = (ground > 0).astype(np.float32)
    return torch.from_numpy(ground).unsqueeze(0).unsqueeze(0).cuda()


def get_inferences(
    image_arrays, model_path, image_paths, paint=False, bin_value=0.5, verbose=0
):
    """
    Obtains the mask predictions of a model for a set of images

    Parameters
    ----------
    image_arrays : array-like
        A list of (1, CH, H, W) images

    image_paths: list(Path)
        A list of paths for images, in the same order as image_arrays

    model_path : str
        The path to a pre-trained model

    Returns
    -------
    masks : list
        A list of (H, W) predicted masks
    """
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)
    to_tensor = ToTensor()

    is_ground = "ground" in Path(model_path).name
    is_instagan = "instagan" in Path(model_path).name

    if is_ground or is_instagan:
        # we just care about he painter here
        ground_path = model_path
        model_path = (
            "/miniscratch/_groups/ccai/experiments/runs/ablation-v1/out--38858350"
        )

    xs = [to_tensor(array).unsqueeze(0) for array in image_arrays]
    xs = [x.to(torch.float32).to(device) for x in xs]
    xs = [(x - 0.5) * 2 for x in xs]
    trainer = Trainer.resume_from_path(
        model_path, inference=True, new_exp=None, device=device
    )
    masks = []
    painted = []
    for idx, x in enumerate(xs):
        if verbose > 0:
            print(idx, "/", len(xs), end="\r")

        if not is_ground and not is_instagan:
            m = trainer.G.mask(x=x)
        else:
            m = load_ground(ground_path, image_paths[idx])

        masks.append(m.squeeze().cpu())
        if paint:
            p = trainer.G.paint(m > bin_value, x)
            painted.append(p.squeeze().cpu())
    return masks, painted


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------
    args = parsed_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    # Determine output dir
    try:
        tmp_dir = Path(os.environ["SLURM_TMPDIR"])
    except Exception as e:
        print(e)
        tmp_dir = Path(input("Enter tmp output directory: ")).resolve()

    plot_dir = tmp_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Build paths to data
    imgs_paths = sorted(
        find_images(args.images_dir, recursive=False), key=lambda x: x.name
    )
    labels_paths = sorted(
        find_images(args.labels_dir, recursive=False),
        key=lambda x: x.name.replace("_labeled.", "."),
    )
    if args.max_files > 0:
        imgs_paths = imgs_paths[: args.max_files]
        labels_paths = labels_paths[: args.max_files]

    print(f"Loading {len(imgs_paths)} images and labels...")

    # Pre-process images: resize + crop
    # TODO: ? make cropping more flexible, not only central
    if not args.prepare_torch:
        ims_labs = [crop_and_resize(i, l) for i, l in zip(imgs_paths, labels_paths)]
        imgs = [d[0] for d in ims_labs]
        labels = [d[1] for d in ims_labs]
    else:
        prepare = PrepareTest()
        imgs = prepare(imgs_paths, normalize=False, rescale=False)
        labels = prepare(labels_paths, normalize=False, rescale=False)

        imgs = [i.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8) for i in imgs]
        labels = [
            lab.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8) for lab in labels
        ]
    imgs = [rgba2rgb(img) if img.shape[-1] == 4 else img for img in imgs]
    print(" Done.")

    # Encode labels
    print("Encode labels...", end="", flush=True)
    # HW label
    labels = [np.squeeze(encode_mask_label(label, "flood")) for label in labels]
    print("Done.")

    if args.yaml:
        y_path = Path(args.yaml)
        assert y_path.exists()
        assert y_path.suffix in {".yaml", ".yml"}
        with y_path.open("r") as f:
            data = yaml.safe_load(f)
        assert "models" in data

        evaluations = [m for m in data["models"]]
    else:
        evaluations = [args.model]

    for e, eval_path in enumerate(evaluations):
        print("\n>>>>> Evaluation", e, ":", eval_path)
        print("=" * 50)
        print("=" * 50)

        model_metrics_path = Path(eval_path) / "eval-metrics"
        model_metrics_path.mkdir(exist_ok=True)
        if args.load_metrics:
            f_csv = model_metrics_path / "eval_masker.csv"
            pred_out = model_metrics_path / "pred"
            if f_csv.exists() and pred_out.exists():
                print("Skipping model because pre-computed metrics exist")
                continue

        # Initialize New Comet Experiment
        exp = Experiment(
            project_name="climategan-masker-metrics", display_summary_level=0
        )

        # Obtain mask predictions
        # TODO: remove (debug)
        print("Obtain mask predictions", end="", flush=True)

        preds, painted = get_inferences(
            imgs,
            eval_path,
            imgs_paths,
            paint=not args.no_paint,
            bin_value=args.bin_value,
            verbose=1,
        )
        preds = [pred.numpy() for pred in preds]
        print(" Done.")

        if args.bin_value > 0:
            preds = [pred > args.bin_value for pred in preds]

        # Compute metrics
        df = pd.DataFrame(
            columns=[
                "tpr",
                "tpt",
                "tnr",
                "tnt",
                "fpr",
                "fpt",
                "fnr",
                "fnt",
                "mnr",
                "mpr",
                "accuracy",
                "error",
                "precision",
                "f05",
                "accuracy_must_may",
                "edge_coherence",
                "filename",
            ]
        )

        print("Compute metrics and plot images")
        for idx, (img, label, pred) in enumerate(zip(*(imgs, labels, preds))):
            print(idx, "/", len(imgs), end="\r")

            # Basic classification metrics
            metrics_dict, maps_dict = masker_classification_metrics(
                pred, label, labels_dict={"cannot": 0, "must": 1, "may": 2}
            )

            # Edges coherence
            edge_coherence, pred_edge, label_edge = edges_coherence_std_min(pred, label)

            series_dict = {
                "tpr": metrics_dict["tpr"],
                "tpt": metrics_dict["tpt"],
                "tnr": metrics_dict["tnr"],
                "tnt": metrics_dict["tnt"],
                "fpr": metrics_dict["fpr"],
                "fpt": metrics_dict["fpt"],
                "fnr": metrics_dict["fnr"],
                "fnt": metrics_dict["fnt"],
                "mnr": metrics_dict["mnr"],
                "mpr": metrics_dict["mpr"],
                "accuracy": metrics_dict["accuracy"],
                "error": metrics_dict["error"],
                "precision": metrics_dict["precision"],
                "f05": metrics_dict["f05"],
                "accuracy_must_may": metrics_dict["accuracy_must_may"],
                "edge_coherence": edge_coherence,
                "filename": str(imgs_paths[idx].name),
            }
            df.loc[idx] = pd.Series(series_dict)

            for k, v in series_dict.items():
                if k == "filename":
                    continue
                exp.log_metric(f"img_{k}", v, step=idx)

            # Confusion matrix
            confmat, _ = get_confusion_matrix(
                metrics_dict["tpr"],
                metrics_dict["tnr"],
                metrics_dict["fpr"],
                metrics_dict["fnr"],
                metrics_dict["mnr"],
                metrics_dict["mpr"],
            )
            confmat = np.around(confmat, decimals=3)
            exp.log_confusion_matrix(
                file_name=imgs_paths[idx].name + ".json",
                title=imgs_paths[idx].name,
                matrix=confmat,
                labels=["Cannot", "Must", "May"],
                row_label="Predicted",
                column_label="Ground truth",
            )

            if args.plot:
                # Plot prediction images
                fig_filename = plot_dir / imgs_paths[idx].name
                plot_images(
                    fig_filename,
                    img,
                    label,
                    pred,
                    metrics_dict,
                    maps_dict,
                    edge_coherence,
                    pred_edge,
                    label_edge,
                )
                exp.log_image(fig_filename)
            if not args.no_paint:
                masked = img * (1 - pred[..., None])
                flooded = img_as_ubyte(
                    (painted[idx].permute(1, 2, 0).cpu().numpy() + 1) / 2
                )
                combined = np.concatenate([img, masked, flooded], 1)
                exp.log_image(combined, imgs_paths[idx].name)

            if args.write_metrics:
                pred_out = model_metrics_path / "pred"
                pred_out.mkdir(exist_ok=True)
                imsave(
                    pred_out / f"{imgs_paths[idx].stem}_pred.png",
                    pred.astype(np.uint8),
                )
                for k, v in maps_dict.items():
                    metric_out = model_metrics_path / k
                    metric_out.mkdir(exist_ok=True)
                    imsave(
                        metric_out / f"{imgs_paths[idx].stem}_{k}.png",
                        v.astype(np.uint8),
                    )

            # --------------------------------
            # -----  END OF IMAGES LOOP  -----
            # --------------------------------

        if args.write_metrics:
            print(f"Writing metrics in {str(model_metrics_path)}")
            f_csv = model_metrics_path / "eval_masker.csv"
            df.to_csv(f_csv, index_label="idx")

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
        exp.log_metrics(dict(means))
        exp.log_table("metrics.csv", df)
        exp.log_html(df.to_html(col_space="80px"))
        exp.log_parameters(vars(args))
        exp.log_parameter("eval_path", str(eval_path))
        exp.add_tag("eval_masker")
        if args.tags:
            exp.add_tags(args.tags)
        exp.log_parameter("model_id", Path(eval_path).name)

        # Close comet
        exp.end()

        # --------------------------------
        # -----  END OF MODElS LOOP  -----
        # --------------------------------

    # Compare models
    if (args.load_metrics or args.write_metrics) and len(evaluations) > 1:
        print(
            "Plots for comparing the input models will be created and logged to comet"
        )

        # Initialize New Comet Experiment
        exp = Experiment(
            project_name="climategan-masker-metrics", display_summary_level=0
        )
        if args.tags:
            exp.add_tags(args.tags)

        # Build DataFrame with all models
        print("Building pandas DataFrame...")
        models_df = {}
        for (m, model_path) in enumerate(evaluations):
            model_path = Path(model_path)
            with open(model_path / "opts.yaml", "r") as f:
                opt = yaml.safe_load(f)
            model_feats = ", ".join(
                [
                    t
                    for t in sorted(opt["comet"]["tags"])
                    if "branch" not in t and "ablation" not in t and "trash" not in t
                ]
            )
            model_id = f"{model_path.parent.name[-2:]}/{model_path.name}"
            df_m = pd.read_csv(
                model_path / "eval-metrics" / "eval_masker.csv", index_col=False
            )
            df_m["model"] = [model_id] * len(df_m)
            df_m["model_idx"] = [m] * len(df_m)
            df_m["model_feats"] = [model_feats] * len(df_m)
            models_df.update({model_id: df_m})
        df = pd.concat(list(models_df.values()), ignore_index=True)
        df["model_img_idx"] = df.model.astype(str) + "-" + df.idx.astype(str)
        df.rename(columns={"idx": "img_idx"}, inplace=True)
        dict_models_labels = {
            k: f"{v['model_idx'][0]}: {v['model_feats'][0]}"
            for k, v in models_df.items()
        }
        print("Done")

        if args.output_csv:
            print(f"Writing DataFrame to {args.output_csv}")
            df.to_csv(args.output_csv, index_label="model_img_idx")

        # Determine images with low metrics in any model
        print("Constructing filter based on metrics thresholds...")
        idx_not_good_in_any = []
        for idx in df.img_idx.unique():
            df_th = df.loc[
                (
                    # TODO: rethink thresholds
                    (df.tpr <= dict_metrics["threshold"]["tpr"])
                    | (df.fpr >= dict_metrics["threshold"]["fpr"])
                    | (df.edge_coherence >= dict_metrics["threshold"]["edge_coherence"])
                )
                & ((df.img_idx == idx) & (df.model.isin(df.model.unique())))
            ]
            if len(df_th) > 0:
                idx_not_good_in_any.append(idx)
        filters = {"all": df.img_idx.unique(), "not_good_in_any": idx_not_good_in_any}
        print("Done")

        # Boxplots of metrics
        print("Plotting boxplots of metrics...")
        for k, f in filters.items():
            print(f"\tDistribution of [{k}] images...")
            for metric in dict_metrics["names"].keys():
                fig_filename = plot_dir / f"boxplot_{metric}_{k}.png"
                if metric in ["mnr", "mpr", "accuracy_must_may"]:
                    boxplot_metric(
                        fig_filename,
                        df.loc[df.img_idx.isin(f)],
                        metric=metric,
                        dict_metrics=dict_metrics["names"],
                        do_stripplot=True,
                        dict_models=dict_models_labels,
                        order=list(df.model.unique()),
                    )
                else:
                    boxplot_metric(
                        fig_filename,
                        df.loc[df.img_idx.isin(f)],
                        metric=metric,
                        dict_metrics=dict_metrics["names"],
                        dict_models=dict_models_labels,
                        fliersize=1.0,
                        order=list(df.model.unique()),
                    )
                exp.log_image(fig_filename)
        print("Done")

        # Cluster Maps
        print("Plotting clustermaps...")
        for k, f in filters.items():
            print(f"\tDistribution of [{k}] images...")
            for metric in dict_metrics["names"].keys():
                fig_filename = plot_dir / f"clustermap_{metric}_{k}.png"
                df_mf = df.loc[df.img_idx.isin(f)].pivot("img_idx", "model", metric)
                clustermap_metric(
                    output_filename=fig_filename,
                    df=df_mf,
                    metric=metric,
                    dict_metrics=dict_metrics["names"],
                    method="average",
                    cluster_metric="euclidean",
                    dict_models=dict_models_labels,
                    row_cluster=False,
                )
                exp.log_image(fig_filename)
        print("Done")

        # Close comet
        exp.end()
