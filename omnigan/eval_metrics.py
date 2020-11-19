import numpy as np
import cv2
import torch
import torch.nn.functional as F

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
    if len(pred.shape) > len(gt_im.shape):
        pred = np.argmax(pred, axis=1)
    return float((pred == gt).sum()) / gt.size


def mIOU(pred, label):
    """
    Adapted from:
    https://stackoverflow.com/questions/62461379/multiclass-semantic-segmentation-model-evaluation

    Compute the mean IOU from pred and label tensors
    pred is a tensor N x C x H x W with logits (softmax will be applied)
    and label is a N x H  x W tensor with int labels per pixel

    Args:
        pred (torch.tensor): predicted logits
        label (torch.tensor): labels

    Returns:
        float: mIOU, can be nan
    """
    num_classes = label.max().item()

    pred = torch.argmax(pred, dim=1).squeeze(1)
    present_iou_list = list()
    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.

    for sem_class in range(num_classes + 1):
        pred_inds = pred == sem_class
        target_inds = label == sem_class
        if target_inds.long().sum().item() > 0:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = (
                pred_inds.long().sum().item()
                + target_inds.long().sum().item()
                - intersection_now
            )
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
    return np.mean(present_iou_list) if present_iou_list else float("nan")
