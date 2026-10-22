#!/usr/bin/env python3
"""Label-free train/valid/test image-domain comparison for KoMaP."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


FEATURES = (
    "gray_mean", "gray_std", "gray_p10", "gray_p90", "gray_entropy_16bin",
    "dark_fraction_lt32", "bright_fraction_ge224",
    "gradient_mean_over_std", "gradient_p90_over_std",
    "laplacian_variance_over_std2", "neighbor_difference_lag1_over_std",
    "neighbor_difference_lag4_over_std", "neighbor_difference_lag16_over_std",
    "gradient_anisotropy", "local_std_mean_over_std",
)
INTENSITY_FEATURES = set(range(7))
GROUP_RE = re.compile(r"^(A\d+)")
STEP_RE = re.compile(r"step\d+")


def image_features(path: Path) -> dict:
    raw = np.asarray(Image.open(path).convert("L"), np.float32)
    height, width = raw.shape
    mean = float(raw.mean())
    std = max(float(raw.std()), 1e-5)
    hist, _ = np.histogram(raw, bins=16, range=(0, 256))
    p = hist.astype(np.float64) / max(int(hist.sum()), 1)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    small = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
    local_var = np.maximum(cv2.blur(small * small, (7, 7)) - cv2.blur(small, (7, 7)) ** 2, 0)
    features = [
        mean, std, float(np.percentile(raw, 10)), float(np.percentile(raw, 90)),
        entropy, float(np.mean(raw < 32)), float(np.mean(raw >= 224)),
        float(gradient.mean() / std), float(np.percentile(gradient, 90) / std),
        float(lap.var() / (std * std)),
    ]
    for lag in (1, 4, 16):
        features.append(float(np.mean(np.abs(small[:, lag:] - small[:, :-lag])) / std))
    energy_x = float(np.mean(gx * gx))
    energy_y = float(np.mean(gy * gy))
    features.append(abs(energy_x - energy_y) / max(energy_x + energy_y, 1e-8))
    features.append(float(np.mean(np.sqrt(local_var)) / std))
    if len(features) != len(FEATURES) or not np.isfinite(features).all():
        raise ValueError(f"invalid image features: {path}")
    name = path.name.removesuffix("_image.png")
    group = GROUP_RE.match(name)
    step = STEP_RE.search(name)
    if not group or not step:
        raise ValueError(f"unexpected image name: {path.name}")
    return {"name": name, "group": group.group(), "step": step.group(),
            "width": width, "height": height, "features": features}


def standardized(all_rows: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    train = np.asarray([r["features"] for r in all_rows["train"]], np.float64)
    center = np.median(train, axis=0)
    q25, q75 = np.percentile(train, [25, 75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, np.std(train, axis=0) * 0.1)
    scale = np.maximum(scale, 1e-6)
    group_weight = np.asarray([1 / np.sqrt(7) if i in INTENSITY_FEATURES else 1 / np.sqrt(8)
                               for i in range(len(FEATURES))])
    return {name: ((np.asarray([r["features"] for r in rows]) - center) / scale) * group_weight
            for name, rows in all_rows.items()}


def nearest(source: np.ndarray, target: np.ndarray, source_rows: list[dict],
            target_rows: list[dict]) -> list[dict]:
    distances = cdist(source, target)
    indices = distances.argmin(axis=1)
    return [{"image": row["name"], "nearest": target_rows[int(indices[i])]["name"],
             "distance": float(distances[i, indices[i]])}
            for i, row in enumerate(source_rows)]


def analyze(root: Path, draws: int) -> dict:
    dirs = {"train": root / "images", "valid": root / "valid_images", "test": root / "test_images"}
    rows = {name: [image_features(path) for path in sorted(folder.glob("*_image.png"))]
            for name, folder in dirs.items()}
    if {name: len(value) for name, value in rows.items()} != {"train": 70, "valid": 20, "test": 10}:
        raise ValueError("expected 70 train, 20 valid, 10 test images")
    z = standardized(rows)
    valid, test = z["valid"], z["test"]
    cross = nearest(test, valid, rows["test"], rows["valid"])
    valid_to_test = nearest(valid, test, rows["valid"], rows["test"])
    a71 = next(row for row in valid_to_test if row["image"] == "A71_step2_sample1")
    all_cross = np.asarray([row["distance"] for row in valid_to_test])
    centroid_distance = float(np.linalg.norm(valid.mean(axis=0) - test.mean(axis=0)))
    pooled = np.concatenate([valid, test])
    random = np.random.default_rng(374)
    greater_or_equal = 0
    for _ in range(draws):
        order = random.permutation(len(pooled))
        delta = np.linalg.norm(pooled[order[:len(valid)]].mean(axis=0)
                               - pooled[order[len(valid):]].mean(axis=0))
        greater_or_equal += delta >= centroid_distance
    p_value = (greater_or_equal + 1) / (draws + 1)
    train_values = np.asarray([r["features"] for r in rows["train"]])
    valid_values = np.asarray([r["features"] for r in rows["valid"]])
    test_values = np.asarray([r["features"] for r in rows["test"]])
    feature_summary = []
    bootstrap_differences = {}
    for i, name in enumerate(FEATURES):
        denom = max(float(np.std(train_values[:, i])), 1e-6)
        feature_summary.append({
            "feature": name,
            "train_mean": float(train_values[:, i].mean()),
            "valid_mean": float(valid_values[:, i].mean()),
            "test_mean": float(test_values[:, i].mean()),
            "test_minus_valid_in_train_standard_deviations":
                float((test_values[:, i].mean() - valid_values[:, i].mean()) / denom),
        })
        if name in ("gray_mean", "gray_std", "dark_fraction_lt32"):
            boot_valid = valid_values[random.integers(0, len(valid_values), size=(draws, len(valid_values))), i].mean(axis=1)
            boot_test = test_values[random.integers(0, len(test_values), size=(draws, len(test_values))), i].mean(axis=1)
            bootstrap_differences[name] = {
                "test_minus_valid": float(test_values[:, i].mean() - valid_values[:, i].mean()),
                "bootstrap_95ci": np.percentile(boot_test - boot_valid, [2.5, 97.5]).tolist(),
            }
    groups = {name: {r["group"] for r in value} for name, value in rows.items()}
    return {
        "scope": "raw images only; test masks and test scores are unavailable",
        "counts": {name: len(value) for name, value in rows.items()},
        "specimen_groups": {name: len(value) for name, value in groups.items()},
        "group_overlap": {
            "train_valid": sorted(groups["train"] & groups["valid"]),
            "train_test": sorted(groups["train"] & groups["test"]),
            "valid_test": sorted(groups["valid"] & groups["test"]),
        },
        "step_counts": {name: dict(Counter(r["step"] for r in value)) for name, value in rows.items()},
        "resolution_counts": {name: dict(Counter(f"{r['width']}x{r['height']}" for r in value))
                              for name, value in rows.items()},
        "feature_summary": feature_summary,
        "bootstrap_feature_differences": bootstrap_differences,
        "valid_test_centroid_distance_train_robust_scaled": centroid_distance,
        "valid_test_permutation_p_value": float(p_value),
        "permutation_draws": draws,
        "a71_nearest_test": a71,
        "a71_valid_to_test_distance_percentile": float(np.mean(all_cross <= a71["distance"])),
        "median_valid_to_test_distance": float(np.median(all_cross)),
        "test_nearest_valid": cross,
        "valid_nearest_test": valid_to_test,
        "interpretation_guard": (
            "Small unlabeled test sample; image features do not prove label or score similarity. "
            "Permutation p-value tests only a difference in the selected feature centroid."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.images_root, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("feature_summary", "test_nearest_valid", "valid_nearest_test")},
                     ensure_ascii=False, indent=2))
    top = sorted(result["feature_summary"],
                 key=lambda row: abs(row["test_minus_valid_in_train_standard_deviations"]),
                 reverse=True)[:6]
    print(json.dumps({"largest_feature_differences": top,
                      "test_nearest_valid": result["test_nearest_valid"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
