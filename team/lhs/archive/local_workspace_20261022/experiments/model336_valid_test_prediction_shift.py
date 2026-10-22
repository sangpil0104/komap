#!/usr/bin/env python3
"""Compare unlabeled valid/test prediction patterns from the same Model336."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
FEATURE_NAMES = (
    *(f"predicted_fraction_{name}" for name in CLASS_NAMES),
    "mean_max_probability", "mean_normalized_entropy",
    "fraction_max_probability_lt_0_6", "predicted_boundary_fraction",
)


def archive_features(path: Path) -> list[dict]:
    rows = []
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            p = np.asarray(archive[name], np.float32)
            if p.ndim != 3 or p.shape[0] != 4 or not np.isfinite(p).all():
                raise ValueError(f"invalid four-class probabilities: {name}")
            if float(np.max(np.abs(p.sum(axis=0) - 1))) > 0.03:
                raise ValueError(f"probabilities are not normalized: {name}")
            pred = p.argmax(axis=0).astype(np.uint8)
            class_fraction = np.bincount(pred.ravel(), minlength=4) / pred.size
            max_p = p.max(axis=0)
            entropy = -np.sum(p * np.log(np.clip(p, 1e-7, 1)), axis=0) / np.log(4)
            boundary = np.zeros(pred.shape, bool)
            boundary[:-1] |= pred[:-1] != pred[1:]
            boundary[:, :-1] |= pred[:, :-1] != pred[:, 1:]
            features = [*class_fraction.tolist(), float(max_p.mean()), float(entropy.mean()),
                        float(np.mean(max_p < 0.6)), float(boundary.mean())]
            rows.append({"image": name, "features": features})
    return rows


def analyze(valid_path: Path, test_path: Path, draws: int) -> dict:
    valid_rows = archive_features(valid_path)
    test_rows = archive_features(test_path)
    if len(valid_rows) != 20 or len(test_rows) != 10:
        raise ValueError("expected 20 valid and 10 test probability maps")
    valid = np.asarray([r["features"] for r in valid_rows])
    test = np.asarray([r["features"] for r in test_rows])
    scale = np.maximum(valid.std(axis=0), 0.02)
    zv = (valid - valid.mean(axis=0)) / scale
    zt = (test - valid.mean(axis=0)) / scale
    distances = cdist(zt, zv)
    nearest = distances.argmin(axis=1)
    a71_index = next(i for i, row in enumerate(valid_rows) if row["image"] == "A71_step2_sample1")
    a71_ranks = [int(np.where(np.argsort(d) == a71_index)[0][0]) + 1 for d in distances]
    observed = float(np.linalg.norm(zv.mean(axis=0) - zt.mean(axis=0)))
    pooled = np.concatenate([zv, zt])
    rng = np.random.default_rng(376)
    exceed = 0
    for _ in range(draws):
        shuffle = rng.permutation(len(pooled))
        delta = np.linalg.norm(pooled[shuffle[:20]].mean(axis=0)
                               - pooled[shuffle[20:]].mean(axis=0))
        exceed += delta >= observed
    bootstrap = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        sampled_valid = valid[rng.integers(0, len(valid), size=(draws, len(valid))), class_id].mean(axis=1)
        sampled_test = test[rng.integers(0, len(test), size=(draws, len(test))), class_id].mean(axis=1)
        bootstrap[class_name] = {
            "test_minus_valid": float(test[:, class_id].mean() - valid[:, class_id].mean()),
            "bootstrap_95ci": np.percentile(sampled_test - sampled_valid, [2.5, 97.5]).tolist(),
        }
    return {
        "scope": "same Model336 checkpoint/inference on valid and test; no test masks used",
        "images": {"valid": len(valid_rows), "test": len(test_rows)},
        "feature_names": list(FEATURE_NAMES),
        "valid_feature_mean": valid.mean(axis=0).tolist(),
        "test_feature_mean": test.mean(axis=0).tolist(),
        "test_minus_valid_feature_mean": (test.mean(axis=0) - valid.mean(axis=0)).tolist(),
        "predicted_class_fraction_bootstrap_differences": bootstrap,
        "valid_test_centroid_permutation_p_value": (exceed + 1) / (draws + 1),
        "permutation_draws": draws,
        "a71_rank_among_20_valid_for_each_test": a71_ranks,
        "test_nearest_valid": [
            {"test_image": row["image"], "nearest_valid": valid_rows[int(nearest[i])]["image"],
             "distance": float(distances[i, nearest[i]])}
            for i, row in enumerate(test_rows)
        ],
        "test_per_image_predicted_fraction": {
            row["image"]: dict(zip(CLASS_NAMES, row["features"][:4])) for row in test_rows
        },
        "valid_images_with_predicted_absent_phase": {
            phase: [r["image"] for r in valid_rows if r["features"][i] == 0]
            for i, phase in enumerate(CLASS_NAMES)
        },
        "test_images_with_predicted_absent_phase": {
            phase: [r["image"] for r in test_rows if r["features"][i] == 0]
            for i, phase in enumerate(CLASS_NAMES)
        },
        "interpretation_guard": "Prediction similarities are model dependent and cannot reveal true test class presence or mIoU.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-probabilities", type=Path, required=True)
    parser.add_argument("--test-probabilities", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.valid_probabilities, args.test_probabilities, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("test_per_image_predicted_fraction", "test_nearest_valid")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
