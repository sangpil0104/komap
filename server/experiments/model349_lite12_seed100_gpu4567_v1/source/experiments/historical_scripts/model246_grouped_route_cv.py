#!/usr/bin/env python3
"""Group-held-out audit of label-free conditional routing for model 245.

The router sees only raw-image / prediction-derived features.  Validation
masks are used solely to fit each fold's route and score held-out A-ID groups.
The official image-wise metric assigns zero to a class with zero union, so all
four class IoUs are averaged for every image.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from improve_five_causes import collect_split
from model215_rich_feature_route import build_features


FEATURES = (
    "gray_mean", "gray_std", "gray_p10", "gray_p50", "gray_p90",
    "gradient_mean", "local_contrast_mean", "entropy_mean", "margin_mean",
    "edge_fraction", "area_al3ni", "area_eutectic",
)
METALLURGICAL_FEATURES = FEATURES + (
    "components_primary", "large_components_primary", "boundary_primary",
    "components_eutectic", "large_components_eutectic", "boundary_eutectic",
    "class_gradient_primary_mean", "class_gradient_eutectic_mean",
)


def specimen_id(path: Path) -> str:
    match = re.match(r"^(A\d+)_", path.name)
    return match.group(1) if match else path.stem


def best_constant(scores: np.ndarray, train: np.ndarray) -> int:
    return int(np.argmax(scores[:, train].mean(axis=1)))


def fit_stump(scores: np.ndarray, features: dict[str, np.ndarray],
              train: np.ndarray, groups: np.ndarray,
              min_images: int, min_groups: int) -> dict:
    base_index = best_constant(scores, train)
    best = {
        "train_score": float(scores[base_index, train].mean()),
        "base_candidate": base_index,
        "feature": None,
        "threshold": None,
        "left_candidate": base_index,
        "right_candidate": base_index,
    }
    for feature_name, values in features.items():
        ordered = np.sort(np.unique(values[train]))
        for low, high in zip(ordered[:-1], ordered[1:]):
            threshold = float((low + high) / 2.0)
            left = train[values[train] < threshold]
            right = train[values[train] >= threshold]
            if len(left) < min_images or len(right) < min_images:
                continue
            if len(np.unique(groups[left])) < min_groups or len(np.unique(groups[right])) < min_groups:
                continue
            left_index = best_constant(scores, left)
            right_index = best_constant(scores, right)
            train_score = float((scores[left_index, left].sum()
                                 + scores[right_index, right].sum()) / len(train))
            if train_score > best["train_score"] + 1e-12:
                best = {
                    "train_score": train_score,
                    "base_candidate": base_index,
                    "feature": feature_name,
                    "threshold": threshold,
                    "left_candidate": left_index,
                    "right_candidate": right_index,
                }
    return best


def describe(row: dict) -> dict:
    keys = ("image_mean_miou", "image_miou_compat", "primary_source", "primary_gamma",
            "al3ni_source", "al3ni_gamma", "eutectic_source", "eutectic_gamma",
            "prior_scale", "beta_scale", "al_delta", "eut_delta")
    return {key: row[key] for key in keys if key in row}


def evaluate_router(scores: np.ndarray, features: dict[str, np.ndarray],
                    groups: np.ndarray, records: list[dict], min_images: int,
                    min_groups: int) -> dict:
    predictions = np.full(scores.shape[1], np.nan, dtype=np.float64)
    fold_records = []
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        route = fit_stump(scores, features, train, groups, min_images, min_groups)
        if route["feature"] is None:
            test_score = scores[route["base_candidate"], test]
        else:
            left = features[route["feature"]][test] < route["threshold"]
            values = np.empty(len(test), dtype=np.float64)
            values[left] = scores[route["left_candidate"], test[left]]
            values[~left] = scores[route["right_candidate"], test[~left]]
            test_score = values
        predictions[test] = test_score
        fold_records.append({
            "held_out_group": str(group),
            "n_images": int(len(test)),
            "train_score": route["train_score"],
            "feature": route["feature"],
            "threshold": route["threshold"],
            "left_candidate": int(route["left_candidate"]),
            "right_candidate": int(route["right_candidate"]),
            "base_candidate": int(route["base_candidate"]),
            "held_out_scores": [float(value) for value in test_score],
        })

    # The comparison baseline is a single, validation-selected model-245 route.
    fixed_index = int(np.argmax(scores.mean(axis=1)))
    fixed = scores[fixed_index]
    per_image_delta = predictions - fixed
    group_deltas = np.asarray([
        per_image_delta[groups == group].mean() for group in np.unique(groups)
    ], dtype=np.float64)
    rng = np.random.default_rng(246)
    boot = rng.choice(group_deltas, size=(10000, len(group_deltas)), replace=True).mean(axis=1)
    return {
        "fixed_candidate_index": fixed_index,
        "fixed_candidate": describe(records[fixed_index]),
        "fixed_image_mean_miou": float(fixed.mean()),
        "grouped_cv_router_image_mean_miou": float(predictions.mean()),
        "grouped_cv_gain_pp": float(per_image_delta.mean() * 100.0),
        "group_bootstrap_95pct_gain_pp": [float(x * 100.0) for x in np.percentile(boot, [2.5, 97.5])],
        "folds": fold_records,
        "oof_scores": [float(value) for value in predictions],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set", choices=("basic", "metallurgical"), default="basic")
    parser.add_argument("--min-images", type=int, default=4)
    parser.add_argument("--min-groups", type=int, default=3)
    args = parser.parse_args()

    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    samples = collect_split(args.data_dir, "valid")
    if len(entries) != len(samples):
        raise ValueError(f"entry/sample count mismatch: {len(entries)} != {len(samples)}")
    features = build_features(entries, samples)
    selected_features = FEATURES if args.feature_set == "basic" else METALLURGICAL_FEATURES
    missing = [name for name in selected_features if name not in features]
    if missing:
        raise ValueError(f"missing features: {missing}")
    payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    records = [row for row in payload["records"]
               if len(row.get("image_scores", [])) == len(samples)]
    if not records:
        raise ValueError("no candidate records with per-image scores")
    scores = np.asarray([row["image_scores"] for row in records], dtype=np.float64)
    groups = np.asarray([specimen_id(image_path) for image_path, _ in samples])

    # In-sample optimum is diagnostic only; the grouped OOF score is the
    # relevant check for whether a condition-based router transfers.
    all_indices = np.arange(scores.shape[1])
    active_features = {key: features[key] for key in selected_features}
    in_sample = fit_stump(scores, active_features, all_indices, groups,
                          args.min_images, args.min_groups)
    cv = evaluate_router(scores, active_features, groups, records, args.min_images,
                         args.min_groups)
    output = {
        "metric": "mean over images of the fixed four class IoUs; empty union -> IoU 0",
        "n_images": int(len(samples)),
        "n_specimen_groups": int(len(np.unique(groups))),
        "candidate_count": int(len(records)),
        "feature_set_name": args.feature_set,
        "feature_set": list(selected_features),
        "in_sample_stump": in_sample,
        "grouped_leave_one_specimen_out": cv,
        "filenames": [image_path.name for image_path, _ in samples],
        "groups": groups.tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_images": output["n_images"],
        "n_groups": output["n_specimen_groups"],
        "candidate_count": output["candidate_count"],
        "fixed_image_mean_miou": cv["fixed_image_mean_miou"],
        "grouped_cv_router_image_mean_miou": cv["grouped_cv_router_image_mean_miou"],
        "grouped_cv_gain_pp": cv["grouped_cv_gain_pp"],
        "group_bootstrap_95pct_gain_pp": cv["group_bootstrap_95pct_gain_pp"],
        "in_sample_stump": in_sample,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
