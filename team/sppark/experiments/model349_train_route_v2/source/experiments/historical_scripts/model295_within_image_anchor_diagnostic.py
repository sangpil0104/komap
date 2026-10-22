#!/usr/bin/env python3
"""Label-free within-image component anchor correction diagnostic.

The fixed rule uses only model probabilities and per-component image statistics
to propose labels. Ground truth is consulted only after proposals are frozen,
for diagnostic metrics and the official per-image mIoU score.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict
from model288_region_classifier_cv import extract_regions, image_score


FEATURE_COLUMNS = np.asarray(
    list(range(3, 14)) + list(range(16, 32)), dtype=np.int64
)


def propose_relabels(features: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, dict]:
    """Propose component relabels from five nearest high-confidence anchors.

    Anchors are predicted secondary components whose mean probability for their
    predicted class is >=0.90 and area is >=max(32 pixels, image median area).
    Features are robust-standardized within the image. Position is excluded.
    An unanchored component changes only when at least four of five nearest
    anchors agree on one class.
    """
    proposed = predicted.copy()
    count = len(predicted)
    if count < 6 or features.shape[1] < 32:
        return proposed, {"anchor_count": 0, "changed_count": 0,
                          "candidate_count": 0, "status": "too_few_components"}

    areas = np.maximum(np.expm1(features[:, 13].astype(np.float64)), 0.0)
    confidence = features[np.arange(count), 3 + predicted]
    anchor_mask = (confidence >= 0.90) & (areas >= max(32.0, float(np.median(areas))))
    anchor_indices = np.flatnonzero(anchor_mask)
    query_indices = np.flatnonzero(~anchor_mask)
    if len(anchor_indices) < 5 or not len(query_indices):
        return proposed, {"anchor_count": int(len(anchor_indices)), "changed_count": 0,
                          "candidate_count": 0, "status": "too_few_anchors_or_queries"}

    raw = np.nan_to_num(features[:, FEATURE_COLUMNS].astype(np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - median), axis=0) * 1.4826
    q25, q75 = np.quantile(raw, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.where(mad > 1e-6, mad, np.where(iqr_scale > 1e-6, iqr_scale, 1.0))
    standardized = np.clip((raw - median) / scale, -8.0, 8.0)

    tree = cKDTree(standardized[anchor_indices])
    _, neighbor_local = tree.query(standardized[query_indices], k=5, workers=1)
    neighbor_classes = predicted[anchor_indices[neighbor_local]]
    votes = np.stack([
        np.sum(neighbor_classes == class_id, axis=1) for class_id in range(4)
    ], axis=1)
    winning_class = votes.argmax(axis=1)
    winning_votes = votes.max(axis=1)
    accepted = (winning_votes >= 4) & (winning_class != predicted[query_indices])
    changed_indices = query_indices[accepted]
    proposed[changed_indices] = winning_class[accepted]

    return proposed, {
        "anchor_count": int(len(anchor_indices)),
        "candidate_count": int(len(query_indices)),
        "changed_count": int(len(changed_indices)),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-baseline", type=float, default=0.78692068)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    started = time.time()

    with Path(args.entries).open("rb") as stream:
        entries = pickle.load(stream)
    with Path(args.settings).open(encoding="utf-8") as stream:
        setting = json.load(stream)["best_image"]
    archive_paths = {
        "195": args.model195, "217": args.tent20,
        "244": args.model244, "249": args.model249,
    }
    archives = {name: np.load(path) for name, path in archive_paths.items()}
    key_orders = [tuple(archive.files) for archive in archives.values()]
    if any(order != key_orders[0] for order in key_orders[1:]):
        raise ValueError("validation probability archive orders differ")
    keys = key_orders[0]
    if len(keys) != len(entries):
        raise ValueError("validation probability archives do not align with cache entries")

    baseline_scores: list[float] = []
    corrected_scores: list[float] = []
    names: list[str] = []
    image_records: list[dict] = []
    totals = {
        "components": 0, "wrong_components": 0, "fixed_wrong_components": 0,
        "correct_components": 0, "contaminated_correct_components": 0,
        "wrong_component_area": 0.0, "fixed_wrong_component_area": 0.0,
        "correct_component_area": 0.0, "contaminated_correct_component_area": 0.0,
    }

    for image_index, (name, item) in enumerate(zip(keys, entries)):
        row = {
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: archive[name].astype(np.float32)
                   for source, archive in archives.items()},
            },
        }
        mixed = make_mixed(row, setting)
        target = item["target"].astype(np.uint8)
        prediction = predict(item, mixed, setting, bilateral_weights(item["z"]), exact=True)
        features, truth, _, predicted, component_ids, component_map = extract_regions(
            prediction, target, mixed, item, connectivity=1)
        proposed, proposal_info = propose_relabels(features, predicted)

        component_areas = np.maximum(np.expm1(features[:, 13].astype(np.float64)), 0.0)
        was_wrong = truth != predicted
        changed = proposed != predicted
        fixed_wrong = was_wrong & changed & (proposed == truth)
        contaminated_correct = (~was_wrong) & changed & (proposed != truth)
        for key, value in (
            ("components", len(truth)),
            ("wrong_components", int(was_wrong.sum())),
            ("fixed_wrong_components", int(fixed_wrong.sum())),
            ("correct_components", int((~was_wrong).sum())),
            ("contaminated_correct_components", int(contaminated_correct.sum())),
            ("wrong_component_area", float(component_areas[was_wrong].sum())),
            ("fixed_wrong_component_area", float(component_areas[fixed_wrong].sum())),
            ("correct_component_area", float(component_areas[~was_wrong].sum())),
            ("contaminated_correct_component_area",
             float(component_areas[contaminated_correct].sum())),
        ):
            totals[key] += value

        output = prediction.copy()
        if len(component_ids):
            class_lookup = np.full(int(component_ids.max()) + 1, 2, dtype=np.uint8)
            class_lookup[component_ids] = proposed.astype(np.uint8)
            active = component_map > 0
            output[active] = class_lookup[component_map[active]]

        base_score = image_score(prediction, target)
        corrected_score = image_score(output, target)
        baseline_scores.append(base_score)
        corrected_scores.append(corrected_score)
        names.append(name)
        image_records.append({
            "name": name, "specimen": name.split("_", 1)[0],
            "baseline": base_score, "corrected": corrected_score,
            "delta_pp": (corrected_score - base_score) * 100.0,
            **proposal_info,
            "component_count": int(len(truth)),
            "wrong_component_count": int(was_wrong.sum()),
            "fixed_wrong_component_count": int(fixed_wrong.sum()),
            "correct_component_count": int((~was_wrong).sum()),
            "contaminated_correct_component_count": int(contaminated_correct.sum()),
        })
        print(f"{name}: base={base_score*100:.4f}% corrected={corrected_score*100:.4f}% "
              f"anchors={proposal_info['anchor_count']} "
              f"changed={proposal_info['changed_count']} "
              f"fixed/wrong={fixed_wrong.sum()}/{was_wrong.sum()} "
              f"contam/correct={contaminated_correct.sum()}/{(~was_wrong).sum()}",
              flush=True)

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    corrected = np.asarray(corrected_scores, dtype=np.float64)
    delta = corrected - baseline
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(name.split("_", 1)[0], []).append(index)
    rng = np.random.default_rng(295)
    bootstrap = np.empty(args.bootstrap_draws, dtype=np.float64)
    group_names = list(groups)
    for draw in range(args.bootstrap_draws):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0

    base_mean = float(baseline.mean())
    if abs(base_mean - args.expected_baseline) > 0.00025:
        raise RuntimeError(
            f"route baseline {base_mean:.8f} differs from expected "
            f"{args.expected_baseline:.8f}; refusing to score mismatched predictions")

    result = {
        "model": 295,
        "method": "fixed label-free within-image high-confidence component anchors",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "ground_truth_use": "diagnostic only after proposals; never used to select anchors or labels",
        "rule": {
            "connectivity": 4,
            "anchor_mean_predicted_class_probability_min": 0.90,
            "anchor_area_min_pixels": 32,
            "anchor_area_min_image_median": True,
            "features": "probability, uncertainty, area, morphology and z/q intensity/texture; no position",
            "nearest_anchor_count": 5,
            "minimum_neighbor_vote": 4,
            "robust_normalization": "within-image median/MAD with IQR fallback",
        },
        "image_count": len(names),
        "specimen_group_count": len(groups),
        "baseline_image_mean_miou": base_mean,
        "corrected_image_mean_miou": float(corrected.mean()),
        "delta_pp": float(delta.mean() * 100.0),
        "positive_image_count": int((delta > 0).sum()),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "component_diagnostics": {
            **{key: int(value) if key in {"components", "wrong_components",
                    "fixed_wrong_components", "correct_components",
                    "contaminated_correct_components"} else float(value)
               for key, value in totals.items()},
            "wrong_component_correction_rate": (
                totals["fixed_wrong_components"] / max(totals["wrong_components"], 1)),
            "wrong_component_area_correction_rate": (
                totals["fixed_wrong_component_area"] / max(totals["wrong_component_area"], 1.0)),
            "correct_component_contamination_rate": (
                totals["contaminated_correct_components"] / max(totals["correct_components"], 1)),
            "correct_component_area_contamination_rate": (
                totals["contaminated_correct_component_area"] /
                max(totals["correct_component_area"], 1.0)),
        },
        "images": image_records,
        "elapsed_seconds": time.time() - started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_image_mean_miou", "corrected_image_mean_miou", "delta_pp",
        "positive_image_count", "group_bootstrap_95ci_delta_pp", "component_diagnostics",
        "elapsed_seconds")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
