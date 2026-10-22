from __future__ import annotations

import numpy as np
from scipy import ndimage

from .common import CLASSES
from .data import boundary_band


def confusion(target, prediction, region=None):
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Expected equal H,W label arrays")
    valid = target != 255
    if region is not None:
        valid &= region
    t, p = target[valid].astype(np.int64), prediction[valid].astype(np.int64)
    if len(t) and (t.min() < 0 or t.max() > 3 or p.min() < 0 or p.max() > 3):
        raise ValueError("Class ID outside 0...3")
    return np.bincount(4 * t + p, minlength=16).reshape(4, 4)


def ious(cm, policy):
    if policy not in ("one", "zero", "exclude"):
        raise ValueError("Invalid empty-union policy")
    tp = cm.diagonal().astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    fill = {"one": 1., "zero": 0., "exclude": np.nan}[policy]
    return np.divide(tp, union, out=np.full(4, fill), where=union > 0)


def average(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def boundary_iou(target, prediction, radius):
    # Class-mask inner boundary bands, including the image frame (zero padding).
    values = []
    for c in range(4):
        t, p = target == c, prediction == c
        def band(mask):
            distance = ndimage.distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
            return mask & (distance <= radius)
        a, b = band(t), band(p)
        union = (a | b).sum()
        values.append(float((a & b).sum() / union) if union else np.nan)
    return values


def measure(target, prediction, cfg):
    cm = confusion(target, prediction)
    row = {"confusion_matrix": cm.tolist()}
    for policy in ("one", "zero", "exclude"):
        row["miou_" + policy] = 100 * average(ious(cm, policy))
    primary = cfg["empty_union"]
    row["image_miou"] = row["miou_" + primary]
    for c, name in enumerate(CLASSES):
        value = ious(cm, primary)[c]
        row["iou_" + name] = float(100 * value) if np.isfinite(value) else None
        row["precision_" + name] = float(cm[c, c] / cm[:, c].sum()) if cm[:, c].sum() else None
        row["recall_" + name] = float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None
    error = target != prediction
    for radius in cfg["boundary_radii"]:
        band = boundary_band(target, radius)
        row[f"boundary_error_{radius}px"] = float(100 * error[band].mean()) if band.any() else None
        values = boundary_iou(target, prediction, radius)
        row[f"boundary_iou_{radius}px"] = 100 * average(values) if average(values) is not None else None
    h, w = target.shape
    yy, xx = np.ogrid[:h, :w]
    for width in cfg["border_widths"]:
        outer = (yy < width) | (yy >= h - width) | (xx < width) | (xx >= w - width)
        for prefix, region in (("outer", outer), ("inner", ~outer)):
            row[f"{prefix}_error_{width}px"] = float(100 * error[region].mean()) if region.any() else None
            value = average(ious(confusion(target, prediction, region), primary)) if region.any() else None
            row[f"{prefix}_miou_{width}px"] = 100 * value if value is not None else None
    return row


def summarize(rows, policy):
    if not rows:
        raise ValueError("No evaluation images")
    out = {"num_images": len(rows), "empty_union": policy,
           "aggregation": "mean over images of class-mean IoU; no global IoU"}
    for key, value in rows[0].items():
        if key not in ("image_id", "confusion_matrix") and (value is None or isinstance(value, (int, float))):
            out[key] = average([r[key] for r in rows])
    return out
