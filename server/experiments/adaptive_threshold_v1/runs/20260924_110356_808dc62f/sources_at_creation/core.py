"""CPU-only, per-image four-level Otsu segmentation and train-only phase mapping."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("Primary Si", "Al3Ni", "Eutectic Si", "Al")
PALETTE = np.array([(76, 178, 76), (25, 76, 153), (204, 204, 204),
                    (153, 127, 76)], dtype=np.uint8)
METHOD = "adaptive_multi_otsu_4_v1"
EMPTY_POLICIES = ("one", "zero", "exclude")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
        if np.any(rgba[..., 3] != 255):
            raise ValueError(f"Transparent pixels are unsupported: {path}")
        return rgba[..., :3].copy()


def gray_from_rgb(rgb):
    # PIL's luminance conversion also supports future RGB input.
    return np.asarray(Image.fromarray(rgb).convert("L"))


def read_target(path):
    rgb = read_rgb(path)
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = class_id
    if np.any(labels == 255):
        unknown = np.unique(rgb[labels == 255], axis=0)[:8].tolist()
        raise ValueError(f"Unknown mask colors in {path}: {unknown}")
    return labels


def multi_otsu(gray):
    """Return three cuts, or None when fewer than four gray values exist.

    Exactly maximize sum_k (sum intensity_k)^2 / count_k over four nonempty
    contiguous histogram intervals. Subtracting the fixed overall mean term
    yields Otsu's between-class variance objective. No labels are consulted.
    """
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a nonempty 2-D uint8 grayscale image")
    histogram = np.bincount(gray.ravel(), minlength=256)
    levels = np.flatnonzero(histogram)
    size = len(levels)
    if size < 4:
        return None
    counts = np.concatenate(([0.0], np.cumsum(histogram[levels], dtype=float)))
    sums = np.concatenate(([0.0], np.cumsum(histogram[levels] * levels, dtype=float)))
    scores = np.full((5, size + 1), -np.inf)
    parent = np.full((5, size + 1), -1, dtype=int)
    scores[0, 0] = 0.0
    for groups in range(1, 5):
        for end in range(groups, size + 1):
            starts = np.arange(groups - 1, end)
            candidates = (scores[groups - 1, starts]
                          + (sums[end] - sums[starts]) ** 2
                          / (counts[end] - counts[starts]))
            best = int(np.argmax(candidates))
            parent[groups, end] = starts[best]
            scores[groups, end] = candidates[best]
    cuts = []
    end = size
    for groups in range(4, 1, -1):
        start = parent[groups, end]
        cuts.append((float(levels[start - 1]) + float(levels[start])) / 2)
        end = start
    return sorted(cuts)


def intensity_bins(gray, thresholds):
    # Lower interval includes the threshold: <=T1, (T1,T2], (T2,T3], >T3.
    return np.searchsorted(thresholds, gray, side="left").astype(np.uint8)


def confusion(target, prediction):
    if target.shape != prediction.shape or not target.size:
        raise ValueError("Nonempty target and prediction shapes must match")
    if np.any(target > 3) or np.any(prediction > 3):
        raise ValueError("Class IDs must be in 0..3")
    return np.bincount(4 * target.astype(np.int64).ravel() + prediction.ravel(),
                       minlength=16).reshape(4, 4)


def class_ious(matrix, empty_policy="one"):
    if empty_policy not in EMPTY_POLICIES:
        raise ValueError(f"Unknown empty-class policy: {empty_policy}")
    tp = np.diag(matrix).astype(float)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - tp
    fill = {"one": 1.0, "zero": 0.0, "exclude": np.nan}[empty_policy]
    values = np.full(4, fill, dtype=float)
    np.divide(tp, union, out=values, where=union > 0)
    return values


def image_metrics(matrix):
    metrics = {}
    for policy in EMPTY_POLICIES:
        ious = class_ious(matrix, policy) * 100
        metrics[f"image_miou_{policy}"] = float(np.nanmean(ious))
        for class_id, value in enumerate(ious):
            metrics[f"class_{class_id}_iou_{policy}"] = (
                float(value) if np.isfinite(value) else None)
    return metrics


def summarize(rows):
    """Average image-level values, never a dataset-wide confusion matrix."""
    if not rows:
        raise ValueError("No image metrics to aggregate")
    summary = {"image_count": len(rows), "unit": "percent", "primary_policy": "one"}
    for policy in EMPTY_POLICIES:
        summary[f"image_miou_{policy}"] = float(np.mean([
            row[f"image_miou_{policy}"] for row in rows]))
        for class_id in range(4):
            key = f"class_{class_id}_iou_{policy}"
            values = [row[key] for row in rows if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
    return summary


def choose_mapping(bin_confusions):
    """Select one global permutation using train image mIoU (empty=one)."""
    if not bin_confusions:
        raise ValueError("Cannot fit phase mapping without training images")
    candidates = []
    for permutation in itertools.permutations(range(4)):
        class_order = np.argsort(permutation)
        score = np.mean([class_ious(cm[:, class_order]).mean()
                         for cm in bin_confusions]) * 100
        candidates.append({"bin_to_class": list(permutation),
                           "train_image_miou_one": float(score)})
    # Iteration order is a deterministic tie-breaker.
    best = max(candidates, key=lambda row: row["train_image_miou_one"])
    return best, candidates


def fit(rows, progress=None):
    if not rows or any(row["source_split"] != "train" for row in rows):
        raise ValueError("Fitting is restricted to nonempty train rows")
    records, matrices, unresolved = [], [], []
    for index, row in enumerate(rows):
        rgb = read_rgb(row["image_path"])
        gray = gray_from_rgb(rgb)
        target = read_target(row["mask_path"])
        if gray.shape != target.shape:
            raise ValueError(f"Image/mask shape mismatch: {row['image_id']}")
        thresholds = multi_otsu(gray)
        matrix = None if thresholds is None else confusion(target, intensity_bins(gray, thresholds))
        if thresholds is None:
            unresolved.append(index)
        records.append({"image_id": row["image_id"], "thresholds": thresholds,
                        "threshold_source": "multi_otsu" if thresholds else "train_fallback",
                        "image_sha256": sha256(row["image_path"]),
                        "mask_sha256": sha256(row["mask_path"])})
        matrices.append(matrix)
        if progress:
            progress(index + 1, row["image_id"])
    normal_cuts = [r["thresholds"] for r in records if r["thresholds"] is not None]
    if not normal_cuts:
        raise ValueError("Every train image has fewer than four gray levels; cannot calibrate")
    fallback = np.median(normal_cuts, axis=0).tolist()
    for index in unresolved:
        gray = gray_from_rgb(read_rgb(rows[index]["image_path"]))
        target = read_target(rows[index]["mask_path"])
        records[index]["thresholds"] = fallback
        matrices[index] = confusion(target, intensity_bins(gray, fallback))
    best, candidates = choose_mapping(matrices)
    for record, matrix in zip(records, matrices):
        record["target_vs_bin_confusion"] = matrix.tolist()
    model = {"method": METHOD, "version": 1, "classes": list(CLASS_NAMES),
             "palette": PALETTE.tolist(), "grayscale": "PIL RGB.convert('L'), uint8",
             "threshold_rule": "per-image four-level Otsu; lower bin includes cut",
             "bin_to_class": best["bin_to_class"], "fallback_thresholds": fallback,
             "empty_class_policy": "one", "selection_metric": "train_image_miou_one",
             "train_image_miou_one": best["train_image_miou_one"],
             "train_image_count": len(rows), "train_images": records}
    return model, candidates


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("method") != METHOD or model.get("version") != 1:
        raise ValueError("Unsupported model format")
    if model.get("classes") != list(CLASS_NAMES) or model.get("palette") != PALETTE.tolist():
        raise ValueError("Model class definitions do not match")
    if sorted(model.get("bin_to_class", [])) != list(range(4)):
        raise ValueError("Invalid bin-to-class permutation")
    cuts = np.asarray(model.get("fallback_thresholds", []), dtype=float)
    if cuts.shape != (3,) or not np.all(np.isfinite(cuts)) or not np.all(np.diff(cuts) > 0):
        raise ValueError("Invalid fallback thresholds")
    return model


def predict(rgb, model):
    gray = gray_from_rgb(rgb)
    thresholds = multi_otsu(gray)
    source = "multi_otsu"
    if thresholds is None:
        thresholds = model["fallback_thresholds"]
        source = "train_fallback"
    mapping = np.asarray(model["bin_to_class"], dtype=np.uint8)
    prediction = mapping[intensity_bins(gray, thresholds)]
    info = {"thresholds": thresholds, "threshold_source": source,
            "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
            "occupied_gray_levels": int(np.count_nonzero(np.bincount(gray.ravel(), minlength=256)))}
    return prediction, info
