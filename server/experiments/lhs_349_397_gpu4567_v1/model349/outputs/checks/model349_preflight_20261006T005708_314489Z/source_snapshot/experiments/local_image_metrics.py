"""Project metric: mean over images, four classes, empty union = zero."""
import numpy as np
from tqdm import tqdm


def class_ious(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected equal H,W label arrays")
    if not (np.isin(prediction, range(4)).all() and np.isin(target, range(4)).all()):
        raise ValueError("Labels must be 0..3")
    intersection = np.array([np.count_nonzero((prediction == c) & (target == c)) for c in range(4)])
    union = np.array([np.count_nonzero((prediction == c) | (target == c)) for c in range(4)])
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def summary(predictions, targets, names):
    rows = [class_ious(p, t) for p, t in zip(predictions, targets, strict=True)]
    if not rows:
        raise ValueError("No images to evaluate")
    values = np.stack(rows)
    scores = values.mean(1)
    score = float(scores.mean())
    classes = dict(zip(names, values.mean(0).tolist()))
    return {
        "image_mean_miou": score, "miou": score,
        "image_miou_compat": score, "image_class_iou": classes, "iou": classes,
        # Historical consumers retain these aliases; all use empty=0 locally.
        "image_mean_miou_exclude": score,
        "image_scores": scores.tolist(), "image_scores_exclude": scores.tolist(),
        "image_mious": scores.tolist(),
        "image_std_miou": float(scores.std()), "image_min_miou": float(scores.min()),
        "image_max_miou": float(scores.max()), "metric": "image_mIoU_four_classes_empty0",
    }


def progress(iterable, desc, total=None):
    return tqdm(iterable, desc=desc, total=total, mininterval=10,
                dynamic_ncols=True, unit="item")
