from PIL import Image
import numpy as np
import cv2

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
    return (pred == gt).sum() / gt.size
