"""
Compute metrics of the performance of the masker using a set of ground-truth labels

python eval_masker.py \
    --model "/miniscratch/schmidtv/vicc/omnigan/runs/predoc7 (4)" \
    --images_dir "/miniscratch/_groups/ccai/data/floodmasks_eval/imgs" \
    --labels_dir "/miniscratch/_groups/ccai/data/floodmasks_eval/labels" \
    --image_size 640 \
    --outputs_dir "/miniscratch/_groups/ccai/data/tmp/metrics" \
"""
print("Imports...", end="")
import os.path
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from omnigan.data import encode_segmap
from omnigan.utils import find_images
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
        "--model",
        required=True,
        type=str,
        help="Path to a pre-trained model",
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
        "--output_dir",
        default="/miniscratch/_groups/ccai/data/omnigan/tmp/metrics/",
        type=str,
        help="DEBUG: Directory containing pre-computed mask predictions",
    )
    parser.add_argument(
        "--image_size",
        default=640,
        type=int,
        help="The height and weight of the pre-processed images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    args = parsed_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    # Build paths to data
    imgs_paths = sorted(find_images(args.images_dir, recursive=False))
    labels_paths = sorted(find_images(args.labels_dir, recursive=False))

    # Pre-process images: resize + crop
    # TODO: ? make cropping more flexible, not only central
    img_preprocessing = PrepareTest(target_size=args.image_size, normalize=False)
    imgs = img_preprocessing(imgs_paths)
    labels = img_preprocessing(labels_paths)

    # Encode labels
    labels = [
        encode_segmap(
            np.squeeze(np.moveaxis(label.numpy().astype(np.uint8), 1, -1)), "flood"
        )
        for label in labels
    ]

    # Obtain mask predictions
    imgs = [img.numpy() for img in imgs]
    if not os.path.isfile(args.model):
        preds_paths = sorted(find_images(args.preds_dir, recursive=False))
        preds = img_preprocessing(preds_paths)
    else:
        preds = get_inferences(imgs, args.model)
    preds = [np.divide(pred.numpy(), np.max(pred.numpy())) for pred in preds]

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
        pred = np.squeeze(pred[:, 0, :, :])

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

    df.to_csv(os.path.join(args.output_dir, "metrics.csv"))
