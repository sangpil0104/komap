from typing import Literal
import numpy as np


def image_miou(
    target: np.ndarray,
    prediction: np.ndarray,
    empty_union: Literal["exclude", "one", "zero"] = "exclude",
    ignore_index: int = 255,
) -> tuple[float, np.ndarray, np.ndarray]:
    if empty_union not in {"exclude", "one", "zero"}:
        raise ValueError(f"Invalid empty-union policy: {empty_union}")
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("Target and prediction must have the same [H,W] shape")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("Target must contain integer class IDs")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError("Prediction must contain integer class IDs")

    valid = target != ignore_index
    if not valid.any():
        raise ValueError("The image has no valid ground-truth pixels")
    t = target[valid].astype(np.int64, copy=False)
    p = prediction[valid].astype(np.int64, copy=False)
    if t.min() < 0 or t.max() > 3 or p.min() < 0 or p.max() > 3:
        raise ValueError("Valid target/prediction IDs must be in [0,3]")

    cm = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = np.full(4, np.nan, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union > 0)
    if empty_union == "one":
        iou[union == 0] = 1.0
    elif empty_union == "zero":
        iou[union == 0] = 0.0
    score = float(np.nanmean(iou)) if empty_union == "exclude" else float(iou.mean())
    return score, iou, cm
