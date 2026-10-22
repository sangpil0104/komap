#!/usr/bin/env python3
"""Prelocked secondary OOF sensitivity for a fixed equal two-seed mean.

This analysis does not alter Model382's primary comparison and never reads
official validation/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from model382_oof_ensemble_compare import (
    GROUP_RE,
    STEP_RE,
    _load_archive,
    _paired_bootstrap,
    _verify_model382_manifest,
    class_ious,
    mask_to_ids,
)


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 3822
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
LOCKED_FILES = (
    "experiments/model382_two_seed_mean_sensitivity_preregistration.md",
    "experiments/model382_two_seed_mean_sensitivity.py",
    "experiments/model382_two_seed_mean_sensitivity_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_lock.py",
    "experiments/model382_seed_count_control_preregistration.md",
    "experiments/model382_five_model_ensemble_addendum.md",
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/results/model382_e5_preregistration_lock.json",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_secondary_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("analysis") != "model382_two_seed_mean_sensitivity_v1":
        raise ValueError("secondary analysis lock has unexpected identity")
    if receipt.get("seed1_oof_scores_read_before_lock") is not False:
        raise ValueError("lock does not attest that seed1 OOF scores were unopened")
    for relative in LOCKED_FILES:
        expected = receipt.get("source_sha256", {}).get(relative)
        path = REPO_ROOT / relative
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"secondary lock/source hash mismatch: {relative}")
    primary_lock = REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json"
    if receipt.get("primary_lock_sha256") != sha256(primary_lock):
        raise ValueError("original Model382 primary lock changed")
    return receipt


def summarize_pairs(rows: dict[str, dict[str, Any]], draws: int = BOOTSTRAP_DRAWS,
                    seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    names = sorted(rows)
    if len(names) != 70:
        raise ValueError(f"expected 70 OOF images, found {len(names)}")
    groups = [rows[name]["group"] for name in names]
    folds = np.asarray([int(rows[name]["fold"]) for name in names], dtype=np.int64)
    arms = ("seed1", "seed2", "mean12")
    per_image_miou = {
        arm: np.asarray([
            float(np.mean(rows[name]["class_iou"][arm])) for name in names
        ], dtype=np.float64)
        for arm in arms
    }
    per_image_class = {
        arm: np.asarray([rows[name]["class_iou"][arm] for name in names], dtype=np.float64)
        for arm in arms
    }

    comparisons: dict[str, dict[str, Any]] = {}
    delta_by_arm: dict[str, np.ndarray] = {}
    fold_by_arm: dict[str, dict[str, float]] = {}
    for reference in ("seed1", "seed2"):
        delta = per_image_miou["mean12"] - per_image_miou[reference]
        delta_by_arm[reference] = delta
        class_delta = (per_image_class["mean12"] - per_image_class[reference]).mean(axis=0) * 100.0
        fold_delta = {
            str(fold): float(delta[folds == fold].mean() * 100.0)
            for fold in range(1, 5)
        }
        fold_by_arm[reference] = fold_delta
        ci = _paired_bootstrap(delta, groups, folds.tolist(), draws, seed)
        comparisons[reference] = {
            "mean12_image_mean_miou_empty0_pct": float(per_image_miou["mean12"].mean() * 100.0),
            "reference_image_mean_miou_empty0_pct": float(per_image_miou[reference].mean() * 100.0),
            "delta_pp": float(delta.mean() * 100.0),
            "group_bootstrap_95_ci_pp": ci,
            "fold_delta_pp": fold_delta,
            "class_delta_pp": {
                class_name: float(value)
                for class_name, value in zip(CLASS_NAMES, class_delta)
            },
            "step4_delta_pp": float(delta[np.asarray([
                int(rows[name]["step"]) == 4 for name in names
            ])].mean() * 100.0),
            "a71_excluded_delta_pp": float(delta[np.asarray([
                rows[name]["group"] != "A71" for name in names
            ])].mean() * 100.0),
        }

    joint_fold_wins = sum(
        fold_by_arm["seed1"][str(fold)] > 0 and fold_by_arm["seed2"][str(fold)] > 0
        for fold in range(1, 5)
    )
    for reference, comparison in comparisons.items():
        comparison["point_delta_at_least_0_20pp"] = comparison["delta_pp"] >= 0.20
        comparison["ci_lower_above_zero"] = comparison["group_bootstrap_95_ci_pp"][0] > 0.0
        comparison["all_class_drops_within_1pp"] = min(comparison["class_delta_pp"].values()) >= -1.0
        comparison["adoption_point_delta_at_least_0_50pp"] = comparison["delta_pp"] >= 0.50
    secondary_gate = (
        all(comparisons[ref]["point_delta_at_least_0_20pp"] for ref in ("seed1", "seed2"))
        and all(comparisons[ref]["ci_lower_above_zero"] for ref in ("seed1", "seed2"))
        and all(comparisons[ref]["all_class_drops_within_1pp"] for ref in ("seed1", "seed2"))
        and joint_fold_wins >= 3
    )
    adoption_gate = (
        secondary_gate
        and all(comparisons[ref]["adoption_point_delta_at_least_0_50pp"]
                for ref in ("seed1", "seed2"))
    )
    return {
        "comparisons": comparisons,
        "folds_where_mean12_beats_both_seeds": joint_fold_wins,
        "joint_fold_majority_gate_passed": joint_fold_wins >= 3,
        "secondary_gate_passed": secondary_gate,
        "adoption_gate_passed": adoption_gate,
    }


def analyze(cv_root: Path, model382_root: Path, lock_path: Path) -> dict[str, Any]:
    receipt = verify_secondary_lock(lock_path)
    original_lock = json.loads(
        (REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json")
        .read_text(encoding="utf-8")
    )
    locked_at = float(original_lock["locked_at_unix"])
    rows: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    group_fold: dict[str, int] = {}
    source_sha: dict[str, str] = {}

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if not masks:
            raise FileNotFoundError(f"missing fixed OOF masks: {mask_dir}")
        expected_ids = set(masks)
        paths: dict[str, Path] = {}
        for model_seed in (1, 2):
            oof_dir = model382_root / f"seed{model_seed}" / f"fold{fold}_oof"
            archive_path = _verify_model382_manifest(oof_dir, model_seed, fold, locked_at)
            paths[f"seed{model_seed}"] = archive_path
            if model_seed == 2:
                manifest_path = oof_dir / "model382_manifest.json"
                if min(archive_path.stat().st_mtime, manifest_path.stat().st_mtime) <= float(receipt["locked_at_unix"]):
                    raise ValueError(f"seed2 OOF artifact predates secondary lock: {oof_dir}")
                checkpoint = model382_root / f"seed2/fold{fold}/last.pt"
                if checkpoint.stat().st_mtime <= float(receipt["locked_at_unix"]):
                    raise ValueError(f"seed2 checkpoint predates secondary lock: {checkpoint}")
        archives = {
            seed: _load_archive(path, expected_ids) for seed, path in paths.items()
        }
        try:
            for image_id in sorted(expected_ids):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image ID across folds: {image_id}")
                seen.add(image_id)
                group_match = GROUP_RE.match(image_id)
                step_match = STEP_RE.search(image_id)
                if not group_match or not step_match:
                    raise ValueError(f"cannot parse group/step from {image_id}")
                group = group_match.group(1)
                previous = group_fold.setdefault(group, fold)
                if previous != fold:
                    raise ValueError(f"group crosses folds: {group}")
                target = mask_to_ids(masks[image_id])
                p1 = np.asarray(archives["seed1"][image_id], dtype=np.float32)
                p2 = np.asarray(archives["seed2"][image_id], dtype=np.float32)
                iou1 = class_ious(p1, target)
                iou2 = class_ious(p2, target)
                iou_mean = class_ious((p1 + p2) * 0.5, target)
                rows[image_id] = {
                    "group": group,
                    "fold": fold,
                    "step": int(step_match.group(1)),
                    "class_iou": {"seed1": iou1, "seed2": iou2, "mean12": iou_mean},
                }
                source_sha[str(masks[image_id])] = sha256(masks[image_id])
        finally:
            for archive in archives.values():
                archive.close()
        for seed, archive_path in paths.items():
            source_sha[str(archive_path)] = sha256(archive_path)
            source_sha[str(archive_path.parent / "model382_manifest.json")] = sha256(
                archive_path.parent / "model382_manifest.json"
            )
            source_sha[str(model382_root / f"seed{seed}/fold{fold}/last.pt")] = sha256(
                model382_root / f"seed{seed}/fold{fold}/last.pt"
            )

    if len(seen) != 70 or len(group_fold) != 59:
        raise ValueError(f"expected exactly 70 images/59 groups; got {len(seen)}/{len(group_fold)}")
    summary = summarize_pairs(rows)
    per_image = {
        image_id: {
            "fold": int(row["fold"]), "group": row["group"], "step": int(row["step"]),
            **{
                arm: [float(value * 100.0) for value in row["class_iou"][arm]]
                for arm in ("seed1", "seed2", "mean12")
            },
        }
        for image_id, row in sorted(rows.items())
    }
    return {
        "model": 382,
        "analysis": "secondary development-only fixed equal-probability two-seed mean",
        "scope": "70 fixed grouped train-OOF images / 59 specimen groups",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "ensemble": "0.5*seed1 probabilities + 0.5*seed2 probabilities, then argmax",
        "primary_protocol_modified": False,
        "official_valid_or_test_read": False,
        "seed1_oof_scores_read_before_secondary_lock": False,
        "secondary_lock_sha256": sha256(lock_path),
        "secondary_locked_at_utc": receipt["locked_at_utc"],
        "bootstrap": {"method": "paired specimen-group, stratified within outer fold",
                      "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "interval": "percentile 95%"},
        **summary,
        "per_image_class_iou_pct": per_image,
        "source_sha256": source_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path,
                        default=REPO_ROOT / "experiments/results/model382_two_seed_mean_preregistration_lock.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.cv_root, args.model382_root, args.lock_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "analysis": result["analysis"],
        "secondary_gate_passed": result["secondary_gate_passed"],
        "adoption_gate_passed": result["adoption_gate_passed"],
        "folds_where_mean12_beats_both_seeds": result["folds_where_mean12_beats_both_seeds"],
        "official_valid_or_test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
