import numpy as np
import cv2
import torch
from skimage import filters
from sklearn.metrics.pairwise import euclidean_distances

# ------------------------------------------------------------------------------
# ----- Evaluation metrics for a pair of binary mask images (pred, target) -----
# ------------------------------------------------------------------------------


def get_accuracy(arr1, arr2):
    """pixel accuracy

    Args:
        arr1 (np.array)
        arr2 (np.array)
    """
    return (arr1 == arr2).sum() / arr1.size


def trimap(pred_im, gt_im, thickness=8):
    """Compute accuracy in a region of thickness around the contours
        for binary images (0-1 values)
    Args:
        pred_im (Image): Prediction
        gt_im (Image): Target
        thickness (int, optional): [description]. Defaults to 8.
    """
    W, H = gt_im.size
    contours, hierarchy = cv2.findContours(
        np.array(gt_im), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    mask_contour = np.zeros((H, W), dtype=np.int32)
    cv2.drawContours(
        mask_contour, contours, -1, (1), thickness=thickness, hierarchy=hierarchy
    )
    gt_contour = np.array(gt_im)[np.where(mask_contour > 0)]
    pred_contour = np.array(pred_im)[np.where(mask_contour > 0)]
    return get_accuracy(pred_contour, gt_contour)


def iou(pred_im, gt_im):
    """
    IoU for binary masks (0-1 values)

    Args:
        pred_im ([type]): [description]
        gt_im ([type]): [description]
    """
    pred = np.array(pred_im)
    gt = np.array(gt_im)
    intersection = (pred * gt).sum()
    union = (pred + gt).sum() - intersection
    return intersection / union


def f1_score(pred_im, gt_im):
    pred = np.array(pred_im)
    gt = np.array(gt_im)
    intersection = (pred * gt).sum()
    return 2 * intersection / (pred + gt).sum()


def accuracy(pred_im, gt_im):
    pred = np.array(pred_im)
    gt = np.array(gt_im)
    if len(gt_im.shape) == 4:
        assert gt_im.shape[1] == 1
        gt_im = gt_im[:, 0, :, :]
    if len(pred.shape) > len(gt_im.shape):
        pred = np.argmax(pred, axis=1)
    return float((pred == gt).sum()) / gt.size


def mIOU(pred, label, average="macro"):
    """
    Adapted from:
    https://stackoverflow.com/questions/62461379/multiclass-semantic-segmentation-model-evaluation

    Compute the mean IOU from pred and label tensors
    pred is a tensor N x C x H x W with logits (softmax will be applied)
    and label is a N x H  x W tensor with int labels per pixel

    this does the same as sklearn's jaccard_score function if you choose average="macro"
    Args:
        pred (torch.tensor): predicted logits
        label (torch.tensor): labels
        average: "macro" or "weighted"

    Returns:
        float: mIOU, can be nan
    """
    num_classes = pred.shape[-3]

    pred = torch.argmax(pred, dim=1).squeeze(1)
    present_iou_list = list()
    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    interesting_classes = (
        [*range(num_classes)] if num_classes > 2 else [int(label.max().item())]
    )
    weights = []

    for sem_class in interesting_classes:
        pred_inds = pred == sem_class
        target_inds = label == sem_class
        if (target_inds.long().sum().item() > 0) or (pred_inds.long().sum().item() > 0):
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = (
                pred_inds.long().sum().item()
                + target_inds.long().sum().item()
                - intersection_now
            )
            weights.append(pred_inds.long().sum().item())
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
    if not present_iou_list:
        return float("nan")
    elif average == "weighted":
        weighted_avg = np.sum(np.multiply(weights, present_iou_list) / np.sum(weights))
        return weighted_avg
    else:
        return np.mean(present_iou_list)


def pred_cannot(pred, label, label_cannot=0):
    """
    Metric for the masker: Computes false positive rate and its map. If the
    predictions are soft, the errors are weighted accordingly.

    Parameters
    ----------
    pred : array-like
        Mask prediction

    label : array-like
        Mask ground truth labels

    label_cannot : int
        The label index of "cannot flood"

    Returns
    -------
    fp_map : array-like
        The map of false positives: predicted mask on cannot flood

    fpr : float
        False positive rate: rate of predicted mask on cannot flood
    """
    fp_map = pred * np.asarray(label == label_cannot, dtype=int)
    fpr = np.sum(fp_map) / np.sum(label == label_cannot)
    return fp_map, fpr


def missed_must(pred, label, label_must=1):
    """
    Metric for the masker: Computes false negative rate and its map. If the
    predictions are soft, the errors are weighted accordingly.

    Parameters
    ----------
    pred : array-like
        Mask prediction

    label : array-like
        Mask ground truth labels

    label_must : int
        The label index of "must flood"

    Returns
    -------
    fn_map : array-like
        The map of false negatives: missed mask on must flood

    fnr : float
        False negative rate: rate of missed mask on must flood
    """
    fn_map = (1.0 - pred) * np.asarray(label == label_must, dtype=int)
    fnr = np.sum(fn_map) / np.sum(label == label_must)
    return fn_map, fnr


def may_flood(pred, label, label_may=2):
    """
    Metric for the masker: Computes "may" negative and "may" positive rates and their
    map. If the predictions are soft, the "errors" are weighted accordingly.

    Parameters
    ----------
    pred : array-like
        Mask prediction

    label : array-like
        Mask ground truth labels

    label_may : int
        The label index of "may flood"

    Returns
    -------
    may_neg_map : array-like
        The map of "may" negatives

    may_pos_map : array-like
        The map of "may" positives

    mnr : float
        "May" negative rate

    mpr : float
        "May" positive rate
    """
    may_neg_map = (1.0 - pred) * np.asarray(label == label_may, dtype=int)
    may_pos_map = pred * np.asarray(label == label_may, dtype=int)
    mnr = np.sum(may_neg_map) / np.sum(label == label_may)
    mpr = np.sum(may_pos_map) / np.sum(label == label_may)
    return may_neg_map, may_pos_map, mnr, mpr


def masker_metrics(pred, label, label_cannot=0, label_must=1):
    """
    Computes a set of metrics for the masker

    Parameters
    ----------
    pred : array-like
        Mask prediction

    label : array-like
        Mask ground truth labels

    label_must : int
        The label index of "must flood"

    label_cannot : int
        The label index of "cannot flood"

    Returns
    -------
    tpr : float
        True positive rate

    tnr : float
        True negative rate

    precision : precision
        Precision, considering only cannot and must flood labels

    f1 : precision
        F1 score, considering only cannot and must flood labels
    """
    tp_map = pred * np.asarray(label == label_must, dtype=int)
    tpr = np.sum(tp_map) / np.sum(label == label_must)
    tn_map = (1.0 - pred) * np.asarray(label == label_cannot, dtype=int)
    tnr = np.sum(tn_map) / np.sum(label == label_cannot)
    fp_map = pred * np.asarray(label == label_cannot, dtype=int)
    fn_map = (1.0 - pred) * np.asarray(label == label_must, dtype=int)
    precision = np.sum(tp_map) / (np.sum(tp_map) + np.sum(fp_map))
    f1 = 2 * (precision * tpr) / (precision + tpr)
    return tpr, tnr, precision, f1


def get_confusion_matrix(tpr, tnr, fpr, fnr, mpr, mnr):
    """
    Constructs the confusion matrix of a masker prediction over a set of samples

    Parameters
    ----------
    tpr : vector-like
        True positive rate

    tnr : vector-like
        True negative rate

    fpr : vector-like
        False positive rate

    fnr : vector-like
        False negative rate

    mpr : vector-like
        "May" positive rate

    mnr : vector-like
        "May" negative rate

    Returns
    -------
    confusion_matrix : 3x3 array
        Confusion matrix: [i, j] = [pred, true]
            | tnr fnr mnr |
            | fpr tpr mpr |
            | 0.  0,  0,  |

    confusion_matrix_std : 3x3 array
        Standard deviation of the confusion matrix
    """
    # Compute mean and standard deviations over all samples
    tpr_m = np.mean(tpr)
    tpr_s = np.std(tpr)
    tnr_m = np.mean(tnr)
    tnr_s = np.std(tnr)
    fpr_m = np.mean(fpr)
    fpr_s = np.std(fpr)
    fnr_m = np.mean(fnr)
    fnr_s = np.std(fnr)
    mpr_m = np.mean(mpr)
    mpr_s = np.std(mpr)
    mnr_m = np.mean(mnr)
    mnr_s = np.std(mnr)

    # Assertions
    assert np.isclose(tpr_m, 1.0 - fnr_m), "TPR: {:.4f}, FNR: {:.4f}".format(tpr_m, fnr_m)
    assert np.isclose(tnr_m, 1.0 - fpr_m), "TNR: {:.4f}, FPR: {:.4f}".format(tnr_m, fpr_m)
    assert np.isclose(mpr_m, 1.0 - mnr_m), "MPR: {:.4f}, MNR: {:.4f}".format(mpr_m, mnr_m)


    # Fill confusion matrix
    confusion_matrix = np.zeros((3, 3))
    confusion_matrix[0, 0] = tnr_m
    confusion_matrix[0, 1] = fnr_m
    confusion_matrix[0, 2] = mnr_m
    confusion_matrix[1, 0] = fpr_m
    confusion_matrix[1, 1] = tpr_m
    confusion_matrix[1, 2] = mpr_m
    confusion_matrix[2, 2] = 0.0

    # Standard deviation
    confusion_matrix_std = np.zeros((3, 3))
    confusion_matrix_std[0, 0] = tnr_s
    confusion_matrix_std[0, 1] = fnr_s
    confusion_matrix_std[0, 2] = mnr_s
    confusion_matrix_std[1, 0] = fpr_s
    confusion_matrix_std[1, 1] = tpr_s
    confusion_matrix_std[1, 2] = mpr_s
    confusion_matrix_std[2, 2] = 0.0
    return confusion_matrix, confusion_matrix_std


def edges_coherence_std_min(pred, label, label_must=1, bin_th=0.5):
    """
    The standard deviation of the minimum distance between the edge of the prediction
    and the edge of the "must flood" label.

    Parameters
    ----------
    pred : array-like
        Mask prediction

    label : array-like
        Mask ground truth labels

    label_must : int
        The label index of "must flood"

    bin_th : float
        The threshold for the binarization of the prediction

    Returns
    -------
    metric : float
        The value of the metric

    pred_edge : array-like
        The edges images of the prediction, for visualization

    label_edge : array-like
        The edges images of the "must flood" label, for visualization
    """
    # Keep must flood label only
    label[label != label_must] = -1
    label[label == label_must] = 1
    label[label != label_must] = 0
    label = np.asarray(label, dtype=float)

    # Binarize prediction
    pred = np.asarray(pred > bin_th, dtype=float)

    # Compute edges
    pred = filters.sobel(pred)
    label = filters.sobel(label)

    # Location of edges
    pred_coord = np.argwhere(pred > 0)
    label_coord = np.argwhere(label > 0)

    # Handle blank predictions
    if pred_coord.shape[0] == 0:
        return 1., pred, label

    # Normalized pairwise distances between pred and label
    dist_mat = np.divide(euclidean_distances(pred_coord, label_coord), pred.shape[0])

    # Standard deviation of the minimum distance from pred to label
    edge_coherence = np.std(np.min(dist_mat, axis=1))

    return edge_coherence, pred, label
