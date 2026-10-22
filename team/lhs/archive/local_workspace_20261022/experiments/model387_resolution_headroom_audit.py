#!/usr/bin/env python3
"""Train-OOF-only all-class boundary-error and oracle-headroom audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import model382_oof_ensemble_compare as model382


CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
BIN_NAMES = ("0_to_lt2px", "2_to_lt4px", "4_to_lt8px", "8px_or_more")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 387
GROUP_RE = re.compile(r"^(A\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_boundary_distance(target: np.ndarray) -> np.ndarray:
    boundary = np.zeros(target.shape, dtype=bool)
    boundary[1:, :] |= target[1:, :] != target[:-1, :]
    boundary[:-1, :] |= target[:-1, :] != target[1:, :]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, :-1] != target[:, 1:]
    if not boundary.any():
        return np.full(target.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def image_miou(target: np.ndarray, prediction: np.ndarray) -> tuple[float, list[float]]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be same-size 2-D class maps")
    if target.size and (target.min() < 0 or target.max() >= 4):
        raise ValueError("target has class IDs outside [0,3]")
    if prediction.size and (prediction.min() < 0 or prediction.max() >= 4):
        raise ValueError("prediction has class IDs outside [0,3]")
    confusion = np.bincount(
        target.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    ious = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
    return float(ious.mean()), ious.astype(float).tolist()


def diagnose_image(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    baseline_miou, baseline_class_iou = image_miou(target, prediction)
    distance = phase_boundary_distance(target)
    bands = (
        distance < 2.0,
        (distance >= 2.0) & (distance < 4.0),
        (distance >= 4.0) & (distance < 8.0),
        distance >= 8.0,
    )
    result: dict[str, Any] = {
        "baseline_image_miou": baseline_miou,
        "baseline_class_iou": baseline_class_iou,
        "bins": {},
    }
    for name, band in zip(BIN_NAMES, bands):
        support = np.bincount(target[band].astype(np.int64), minlength=4)
        false_negative = np.bincount(
            target[band & (prediction != target)].astype(np.int64), minlength=4
        )
        rates = [
            float(false_negative[c] / support[c]) if support[c] else None
            for c in range(4)
        ]
        oracle_prediction = prediction.copy()
        oracle_prediction[band] = target[band]
        oracle_miou, oracle_class_iou = image_miou(target, oracle_prediction)
        result["bins"][name] = {
            "pixels": int(band.sum()),
            "wrong_pixels": int(np.count_nonzero(band & (prediction != target))),
            "gt_support_by_class": support.astype(int).tolist(),
            "fn_by_class": false_negative.astype(int).tolist(),
            "fn_rate_by_class": rates,
            "oracle_image_miou": oracle_miou,
            "oracle_class_iou": oracle_class_iou,
            "oracle_miou_gain": float(oracle_miou - baseline_miou),
        }
    return result


def _pool_class_rate(rows: list[dict[str, Any]], class_id: int, indices: tuple[int, ...]) -> float | None:
    support = sum(
        row["diagnostic"]["bins"][BIN_NAMES[index]]["gt_support_by_class"][class_id]
        for row in rows for index in indices
    )
    misses = sum(
        row["diagnostic"]["bins"][BIN_NAMES[index]]["fn_by_class"][class_id]
        for row in rows for index in indices
    )
    return float(misses / support) if support else None


def _stratified_group_bootstrap(
    rows: list[dict[str, Any]], folds: tuple[int, int], draws: int
) -> list[list[dict[str, Any]]]:
    by_fold_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_fold_group[row["fold"]][row["specimen_group"]].append(row)
    for fold in folds:
        if len(by_fold_group[fold]) < 2:
            raise ValueError(f"fold {fold} needs at least two specimen groups for bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[list[dict[str, Any]]] = []
    for _ in range(draws):
        selected: list[dict[str, Any]] = []
        for fold in folds:
            groups = sorted(by_fold_group[fold])
            for index in rng.integers(0, len(groups), size=len(groups)):
                selected.extend(by_fold_group[fold][groups[int(index)]])
        samples.append(selected)
    return samples


def _ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5]).tolist()


def summarize(rows: list[dict[str, Any]], folds: tuple[int, int], draws: int) -> dict[str, Any]:
    samples = _stratified_group_bootstrap(rows, folds, draws)
    oracle: dict[str, Any] = {}
    for bin_name in BIN_NAMES:
        observed = float(np.mean([
            row["diagnostic"]["bins"][bin_name]["oracle_miou_gain"] for row in rows
        ]))
        boot = [float(np.mean([
            row["diagnostic"]["bins"][bin_name]["oracle_miou_gain"] for row in sample
        ])) for sample in samples]
        oracle[bin_name] = {
            "mean_imagewise_miou_gain": observed,
            "group_bootstrap_95ci": _ci(boot),
            "interpretation": "perfect-correction upper bound, not expected achievable gain",
        }

    class_rates: dict[str, Any] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        near = _pool_class_rate(rows, class_id, (0, 1))
        interior = _pool_class_rate(rows, class_id, (3,))
        observed_delta = near - interior if near is not None and interior is not None else None
        boot_delta: list[float] = []
        for sample in samples:
            sample_near = _pool_class_rate(sample, class_id, (0, 1))
            sample_interior = _pool_class_rate(sample, class_id, (3,))
            if sample_near is not None and sample_interior is not None:
                boot_delta.append(float(sample_near - sample_interior))
        by_bin: dict[str, Any] = {}
        for bin_index, bin_name in enumerate(BIN_NAMES):
            support = sum(row["diagnostic"]["bins"][bin_name]["gt_support_by_class"][class_id]
                          for row in rows)
            fn = sum(row["diagnostic"]["bins"][bin_name]["fn_by_class"][class_id]
                     for row in rows)
            by_bin[bin_name] = {
                "gt_support": int(support),
                "false_negatives": int(fn),
                "fn_rate": float(fn / support) if support else None,
            }
        class_rates[class_name] = {
            "distance_bins": by_bin,
            "near_0_to_lt4px_fn_rate": near,
            "interior_ge8px_fn_rate": interior,
            "near_minus_interior_rate_difference": observed_delta,
            "near_minus_interior_group_bootstrap_95ci": _ci(boot_delta),
        }

    per_fold: dict[str, Any] = {}
    for fold in folds:
        fold_rows = [row for row in rows if row["fold"] == fold]
        per_fold[str(fold)] = {
            "images": len(fold_rows),
            "groups": len({row["specimen_group"] for row in fold_rows}),
            "mean_imagewise_miou": float(np.mean([
                row["diagnostic"]["baseline_image_miou"] for row in fold_rows
            ])),
            "oracle_gain_by_bin": {
                bin_name: float(np.mean([
                    row["diagnostic"]["bins"][bin_name]["oracle_miou_gain"]
                    for row in fold_rows
                ])) for bin_name in BIN_NAMES
            },
        }
    return {
        "images": len(rows),
        "groups": len({row["specimen_group"] for row in rows}),
        "mean_imagewise_miou_empty_union_zero": float(np.mean([
            row["diagnostic"]["baseline_image_miou"] for row in rows
        ])),
        "official_metric": "per image: mean of 4 class IoUs; empty union=0; then mean over images",
        "class_distance_rates": class_rates,
        "oracle_miou_gain_by_bin": oracle,
        "per_fold": per_fold,
        "bootstrap": {
            "method": "resample specimen groups with replacement within each included outer fold",
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "unit": "specimen group; all images in each selected group retained",
        },
    }


def _verify_lock_receipt(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected_hash in receipt["artifacts_sha256"].items():
        path = model382.REPO_ROOT / relative
        if not path.is_file() or model382.sha256(path) != expected_hash:
            raise ValueError(f"Model382 preregistration lock mismatch: {relative}")
    return receipt


def analyze(cv_root: Path, model382_root: Path, folds: tuple[int, int],
            lock_receipt: Path, draws: int) -> dict[str, Any]:
    if folds not in ((1, 2), (3, 4)):
        raise ValueError("run exploratory folds (1,2) or pre-registered diagnostic folds (3,4)")
    receipt = _verify_lock_receipt(lock_receipt)
    seed_results: dict[str, Any] = {}
    all_sources: dict[str, str] = {}

    for seed in (1, 2):
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        group_fold: dict[str, int] = {}
        for fold in folds:
            mask_dir = cv_root / f"oof_fold{fold}" / "masks"
            oof_dir = model382_root / f"seed{seed}" / f"fold{fold}_oof"
            archive_path = model382._verify_model382_manifest(
                oof_dir, seed, fold, float(receipt["locked_at_unix"])
            )
            manifest_path = oof_dir / "model382_manifest.json"
            masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
            if not masks:
                raise FileNotFoundError(f"no fixed-fold masks: {mask_dir}")
            with np.load(archive_path, allow_pickle=False) as archive:
                if set(archive.files) != set(masks):
                    raise ValueError(f"OOF IDs differ from fixed-fold masks: {archive_path}")
                for image_id in sorted(masks):
                    if image_id in seen_ids:
                        raise ValueError(f"duplicate OOF image across folds: {image_id}")
                    seen_ids.add(image_id)
                    match = GROUP_RE.match(image_id)
                    if not match:
                        raise ValueError(f"cannot parse specimen group from {image_id}")
                    group = match.group(1)
                    if group in group_fold and group_fold[group] != fold:
                        raise ValueError(f"group crosses OOF folds: {group}")
                    group_fold[group] = fold
                    target = model382.mask_to_ids(masks[image_id])
                    probabilities = np.asarray(archive[image_id], dtype=np.float32)
                    if probabilities.shape != (4, *target.shape):
                        raise ValueError(f"bad OOF probability shape for {image_id}: {probabilities.shape}")
                    if not np.isfinite(probabilities).all():
                        raise ValueError(f"non-finite OOF probabilities for {image_id}")
                    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
                        raise ValueError(f"OOF probability range invalid for {image_id}")
                    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"OOF probabilities are not normalized for {image_id}")
                    prediction = probabilities.argmax(axis=0).astype(np.uint8)
                    rows.append({
                        "image_id": image_id,
                        "specimen_group": group,
                        "fold": fold,
                        "diagnostic": diagnose_image(target, prediction),
                    })
                    all_sources[str(masks[image_id])] = sha256(masks[image_id])
            all_sources[str(archive_path)] = sha256(archive_path)
            all_sources[str(manifest_path)] = sha256(manifest_path)
            checkpoint = oof_dir.parent / f"fold{fold}" / "last.pt"
            all_sources[str(checkpoint)] = sha256(checkpoint)

        seed_results[str(seed)] = summarize(rows, folds, draws)
        seed_results[str(seed)]["per_image"] = rows

    primary = "confirmatory-style diagnostic subset; folds3-4 were excluded from Model386 hypothesis generation" \
        if folds == (3, 4) else "exploratory only; folds1-2 informed Model385/386 hypotheses"
    return {
        "model": 387,
        "scope": primary,
        "folds": list(folds),
        "seed_results": seed_results,
        "model382_preregistration_lock": {
            "path": str(lock_receipt),
            "locked_at_utc": receipt.get("locked_at_utc"),
            "sha256": sha256(lock_receipt),
        },
        "source_sha256": all_sources,
        "official_valid_or_test_read": False,
        "training_started": False,
        "automatic_resolution_training_go": False,
        "interpretation_guard": (
            "Train-OOF diagnostic only. Boundary oracle gains are unattainable upper bounds, not forecasts. "
            "A positive boundary pattern does not identify feature stride as its cause. "
            "Any resolution intervention needs a separate preregistered, group-aware comparison."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs=2, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--lock-receipt", type=Path,
                        default=Path("experiments/results/model382_e5_preregistration_lock.json"))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise SystemExit("--bootstrap-draws must be >=100")
    result = analyze(args.cv_root, args.model382_root, tuple(sorted(args.folds)),
                     args.lock_receipt, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": result["model"],
        "scope": result["scope"],
        "folds": result["folds"],
        "seed_results": {
            seed: {key: value for key, value in metrics.items() if key != "per_image"}
            for seed, metrics in result["seed_results"].items()
        },
        "training_started": False,
        "official_valid_or_test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
