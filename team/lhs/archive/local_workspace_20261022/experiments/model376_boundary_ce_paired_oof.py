#!/usr/bin/env python3
"""Paired, specimen-grouped OOF comparison for Model376 boundary-weighted CE.

Only train-fold OOF probabilities and their corresponding train masks are read.
Official validation and test inputs are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from model374_train_oof_error_diagnostic import mask_to_ids


GROUP_RE = re.compile(r"^(A\d+)")
CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]


def image_iou(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    if probabilities.shape != (4, *target.shape):
        raise ValueError(f"probability/mask shape mismatch: {probabilities.shape} vs {target.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite probability value")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError("probabilities are not normalized")
    prediction = probabilities.argmax(axis=0).astype(np.int64)
    truth = target.astype(np.int64)
    matrix = np.bincount(truth.ravel() * 4 + prediction.ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def load_scores(cv_data: Path, oof_root: Path, label: str) -> dict[str, dict]:
    records = {}
    for fold in range(1, 5):
        mask_dir = cv_data / f"oof_fold{fold}" / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        masks = {path.name.removesuffix("_mask.png"): path for path in mask_dir.glob("*_mask.png")}
        if not masks or not archive_path.is_file():
            raise FileNotFoundError(f"missing fold-{fold} OOF masks or {label} probabilities")
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: {label} prediction/mask IDs do not match")
            for name in sorted(masks):
                match = GROUP_RE.match(name)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {name}")
                scores = image_iou(np.asarray(archive[name], dtype=np.float32), mask_to_ids(masks[name]))
                if name in records:
                    raise ValueError(f"duplicate OOF image: {name}")
                records[name] = {"fold": fold, "group": match.group(1), "class_iou": scores}
    return records


def compare(cv_data: Path, baseline_root: Path, candidate_root: Path, draws: int) -> dict:
    baseline = load_scores(cv_data, baseline_root, "Model374")
    candidate = load_scores(cv_data, candidate_root, "Model376")
    if baseline.keys() != candidate.keys() or len(baseline) != 70:
        raise ValueError(f"OOF coverage mismatch: baseline={len(baseline)}, candidate={len(candidate)}")

    names = sorted(baseline)
    groups = [baseline[name]["group"] for name in names]
    base_iou = np.stack([baseline[name]["class_iou"] for name in names])
    cand_iou = np.stack([candidate[name]["class_iou"] for name in names])
    base_score = base_iou.mean(axis=1)
    cand_score = cand_iou.mean(axis=1)
    delta = cand_score - base_score
    group_indices: dict[str, np.ndarray] = {
        group: np.flatnonzero(np.asarray(groups) == group) for group in sorted(set(groups))
    }
    group_names = list(group_indices)
    per_group = {
        group: float(np.mean(delta[indices])) for group, indices in group_indices.items()
    }
    per_image = [{
        "image_id": name,
        "specimen_group": groups[i],
        "fold": candidate[name]["fold"],
        "model374_image_miou_empty0": float(base_score[i]),
        "model376_image_miou_empty0": float(cand_score[i]),
        "delta_percentage_points": float(delta[i] * 100.0),
        "model374_class_iou": base_iou[i].tolist(),
        "model376_class_iou": cand_iou[i].tolist(),
    } for i, name in enumerate(names)]

    rng = np.random.default_rng(376)
    group_draws = rng.integers(0, len(group_names), size=(draws, len(group_names)))
    boot_delta = np.empty(draws, dtype=np.float64)
    boot_class_delta = np.empty((draws, 4), dtype=np.float64)
    for draw_i, sample in enumerate(group_draws):
        image_indices = np.concatenate([group_indices[group_names[index]] for index in sample])
        boot_delta[draw_i] = float(delta[image_indices].mean())
        boot_class_delta[draw_i] = (cand_iou[image_indices] - base_iou[image_indices]).mean(axis=0)

    folds = {}
    for fold in range(1, 5):
        fold_indices = np.asarray([i for i, name in enumerate(names) if candidate[name]["fold"] == fold])
        folds[str(fold)] = {
            "images": int(len(fold_indices)),
            "model374_image_miou_empty0": float(base_score[fold_indices].mean()),
            "model376_image_miou_empty0": float(cand_score[fold_indices].mean()),
            "delta_percentage_points": float(delta[fold_indices].mean() * 100.0),
        }

    lower, upper = np.percentile(boot_delta * 100.0, [2.5, 97.5])
    class_ci = {
        name: np.percentile(boot_class_delta[:, i] * 100.0, [2.5, 97.5]).tolist()
        for i, name in enumerate(CLASS_NAMES)
    }
    positive_folds = sum(row["delta_percentage_points"] > 0 for row in folds.values())
    max_regression = min(row["delta_percentage_points"] for row in folds.values())
    gate = {
        "all_oof_gain_at_least_0_5pp": float(delta.mean() * 100.0) >= 0.5,
        "at_least_3_of_4_folds_improve": positive_folds >= 3,
        "no_fold_regresses_over_0_3pp": max_regression >= -0.3,
        "group_bootstrap_95ci_lower_above_zero": float(lower) > 0,
    }
    return {
        "model": 376,
        "scope": "70 train images, exactly-once specimen-grouped OOF; no valid or test labels",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "images": len(names),
        "specimen_groups": len(group_names),
        "model374_image_miou_empty0": float(base_score.mean()),
        "model376_image_miou_empty0": float(cand_score.mean()),
        "delta_percentage_points": float(delta.mean() * 100.0),
        "group_bootstrap_95ci_delta_percentage_points": [float(lower), float(upper)],
        "class_mean_iou_delta_percentage_points": {
            name: float((cand_iou[:, i].mean() - base_iou[:, i].mean()) * 100.0)
            for i, name in enumerate(CLASS_NAMES)
        },
        "class_group_bootstrap_95ci_delta_percentage_points": class_ci,
        "improved_images": int((delta > 0).sum()),
        "improved_specimen_groups": int(sum(value > 0 for value in per_group.values())),
        "folds": folds,
        "adoption_gate": gate,
        "adoption_gate_passed": all(gate.values()),
        "per_image": per_image,
        "per_specimen_group_delta_percentage_points": {
            key: value * 100.0 for key, value in per_group.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.cv_data, args.baseline_root, args.candidate_root, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = {key: value for key, value in result.items() if key not in (
        "per_image", "per_specimen_group_delta_percentage_points"
    )}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
