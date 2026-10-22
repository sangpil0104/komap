#!/usr/bin/env python3
"""Model389 CPU audits: exact training-GT dose and grouped-OOF headroom."""

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
from PIL import Image


PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
MINORITY_CLASSES = (1, 3)
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
AREA_BINS = (
    (1, 64, "1_to_lt64"),
    (64, 256, "64_to_lt256"),
    (256, 1024, "256_to_lt1024"),
    (1024, 4096, "1024_to_lt4096"),
    (4096, None, "4096_or_more"),
)
EPOCHS = 70
SAMPLES_PER_IMAGE = 12
POSITIVE_PROBABILITY = 0.5
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 389
GROUP_RE = re.compile(r"^(A\d+)_")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown mask colors in {path}: {np.unique(rgb[ids == 255], axis=0)[:8].tolist()}")
    return ids


def component_areas(mask: np.ndarray, class_id: int) -> np.ndarray:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask == class_id).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    return stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if count > 1 else np.zeros(0, dtype=np.int64)


def area_bin(area: int) -> str:
    for low, high, name in AREA_BINS:
        if area >= low and (high is None or area < high):
            return name
    raise ValueError(f"invalid component area: {area}")


def load_fold_masks(cv_root: Path) -> tuple[dict[int, dict[str, Path]], dict[str, int]]:
    masks_by_fold: dict[int, dict[str, Path]] = {}
    group_fold: dict[str, int] = {}
    seen: set[str] = set()
    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"missing train-fold masks: {mask_dir}")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if not masks:
            raise FileNotFoundError(f"empty train-fold mask directory: {mask_dir}")
        masks_by_fold[fold] = masks
        for image_id in masks:
            if image_id in seen:
                raise ValueError(f"duplicate image across fixed OOF folds: {image_id}")
            seen.add(image_id)
            match = GROUP_RE.match(image_id)
            if not match:
                raise ValueError(f"cannot derive specimen group: {image_id}")
            group = match.group(1)
            if group in group_fold and group_fold[group] != fold:
                raise ValueError(f"specimen group crosses folds: {group}")
            group_fold[group] = fold
    return masks_by_fold, group_fold


def expected_component_anchors(areas: np.ndarray, positive_budget: float) -> dict[str, np.ndarray]:
    total = int(areas.sum())
    if total == 0:
        return {"current": np.zeros(0), "model388": np.zeros(0), "model389": np.zeros(0)}
    tiny = areas < 64
    eligible = ~tiny
    tiny_mass = int(areas[tiny].sum())
    n_eligible = int(eligible.sum())
    current = positive_budget * areas.astype(np.float64) / total
    if n_eligible:
        v388 = np.where(eligible, positive_budget / n_eligible, 0.0)
        eligible_budget = positive_budget * (total - tiny_mass) / total
        v389 = np.where(tiny, current, eligible_budget / n_eligible)
    else:
        v388 = current.copy()
        v389 = current.copy()
    return {"current": current, "model388": v388, "model389": v389}


def exposure_audit(cv_root: Path, folds: tuple[int, int]) -> dict[str, Any]:
    if folds not in ((1, 2), (3, 4)):
        raise ValueError("run folds (1,2) exploratory or folds (3,4) primary")
    masks_by_fold, group_fold = load_fold_masks(cv_root)
    per_outer: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for heldout_fold in folds:
        records: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        class_totals: dict[int, dict[str, float]] = {
            c: {"current": 0.0, "model388": 0.0, "model389": 0.0} for c in MINORITY_CLASSES
        }
        per_image_checks: list[dict[int, dict[str, float]]] = []
        fallback_image_class_pairs = 0
        minority_image_class_pairs = 0
        train_ids: set[str] = set()
        for source_fold in range(1, 5):
            if source_fold == heldout_fold:
                continue
            for image_id, mask_path in sorted(masks_by_fold[source_fold].items()):
                train_ids.add(image_id)
                sources[str(mask_path)] = sha256(mask_path)
                mask = read_mask(mask_path)
                areas_by_class = {c: component_areas(mask, c) for c in MINORITY_CLASSES}
                present = [c for c in MINORITY_CLASSES if areas_by_class[c].size]
                if not present:
                    continue
                budget = EPOCHS * SAMPLES_PER_IMAGE * POSITIVE_PROBABILITY / len(present)
                group = GROUP_RE.match(image_id).group(1)
                image_class_totals: dict[int, dict[str, float]] = {}
                for class_id in present:
                    areas = areas_by_class[class_id]
                    minority_image_class_pairs += 1
                    if not np.any(areas >= 64):
                        fallback_image_class_pairs += 1
                    expected = expected_component_anchors(areas, budget)
                    image_class_totals[class_id] = {}
                    for sampler in ("current", "model388", "model389"):
                        image_class_totals[class_id][sampler] = float(expected[sampler].sum())
                        class_totals[class_id][sampler] += float(expected[sampler].sum())
                    for index, area in enumerate(areas.tolist()):
                        records[(class_id, area_bin(int(area)))].append({
                            "image_id": image_id,
                            "specimen_group": group,
                            "area": int(area),
                            "current": float(expected["current"][index]),
                            "model388": float(expected["model388"][index]),
                            "model389": float(expected["model389"][index]),
                        })
                per_image_checks.append(image_class_totals)

        class_result: dict[str, Any] = {}
        for class_id in MINORITY_CLASSES:
            bins: dict[str, Any] = {}
            for _, _, name in AREA_BINS:
                items = records[(class_id, name)]
                sums = {key: float(sum(item[key] for item in items)) for key in ("current", "model388", "model389")}
                ratios = [item["model389"] / item["current"] for item in items if item["current"] > 0]
                bins[name] = {
                    "components": len(items),
                    "groups": len({item["specimen_group"] for item in items}),
                    "current_expected_anchors": sums["current"],
                    "model388_expected_anchors": sums["model388"],
                    "model389_expected_anchors": sums["model389"],
                    "model388_to_current_ratio": sums["model388"] / sums["current"] if sums["current"] else None,
                    "model389_to_current_ratio": sums["model389"] / sums["current"] if sums["current"] else None,
                    "model389_component_dose_ratio_median": float(np.median(ratios)) if ratios else None,
                    "model389_component_dose_ratio_below1_share": (
                        float(np.mean(np.asarray(ratios) < 1.0)) if ratios else None
                    ),
                }
            totals = class_totals[class_id]
            class_result[CLASS_NAMES[class_id]] = {
                "total_expected_anchors": totals,
                "class_total_delta_model388": totals["model388"] - totals["current"],
                "class_total_delta_model389": totals["model389"] - totals["current"],
                "area_bins": bins,
            }
        per_outer[str(heldout_fold)] = {
            "training_images": len(train_ids),
            "training_groups": len({GROUP_RE.match(name).group(1) for name in train_ids}),
            "heldout_fold_excluded": heldout_fold,
            "minority_image_class_pairs": minority_image_class_pairs,
            "fallback_image_class_pairs_without_eligible_component": fallback_image_class_pairs,
            "fallback_pair_share": (
                fallback_image_class_pairs / minority_image_class_pairs
                if minority_image_class_pairs else None
            ),
            "max_absolute_per_image_class_total_anchor_delta_model388": max((
                abs(totals[class_id]["model388"] - totals[class_id]["current"])
                for totals in per_image_checks for class_id in totals
            ), default=0.0),
            "max_absolute_per_image_class_total_anchor_delta_model389": max((
                abs(totals[class_id]["model389"] - totals[class_id]["current"])
                for totals in per_image_checks for class_id in totals
            ), default=0.0),
            "classes": class_result,
        }
    return {
        "model": 389,
        "mode": "E1 exact mass-preserving expected anchor dose; training GT only",
        "scope": "primary folds3-4" if folds == (3, 4) else "exploratory folds1-2",
        "folds": list(folds),
        "total_images": sum(len(v) for v in masks_by_fold.values()),
        "total_groups": len(group_fold),
        "minority_classes": [CLASS_NAMES[c] for c in MINORITY_CLASSES],
        "sampling_budget": {
            "epochs": EPOCHS,
            "samples_per_image_per_epoch": SAMPLES_PER_IMAGE,
            "positive_probability": POSITIVE_PROBABILITY,
            "crop_native_px": 256,
            "jitter_each_axis_px": 64,
            "coordinate_cap": 4000,
        },
        "formula": {
            "tiny_component_model389": "B * area / total_class_pixels (exactly current pixel sampler)",
            "eligible_component_model389": "B * eligible_class_pixels / (total_class_pixels * eligible_component_count)",
        },
        "per_outer_fold": per_outer,
        "source_sha256": sources,
        "official_valid_or_test_read": False,
        "oof_values_read": False,
        "interpretation_guard": "Expected anchor-dose redistribution only; no performance or causal claim.",
    }


def image_miou(target: np.ndarray, prediction: np.ndarray) -> float:
    confusion = np.bincount(
        target.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(), minlength=16
    ).reshape(4, 4)
    inter = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - inter
    iou = np.divide(inter, union, out=np.zeros(4), where=union > 0)
    return float(iou.mean())


def image_outcomes(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    base = image_miou(target, prediction)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (target == 3).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0, dtype=np.int64)
    tgt_misses = tgt_components = cmp_misses = cmp_components = 0
    target_mask = np.zeros(target.shape, dtype=bool)
    for component_idx, area in enumerate(areas, start=1):
        component = labels == component_idx
        recall = float(np.count_nonzero(prediction[component] == 3) / area)
        missed = recall < 0.5
        if 64 <= area < 256:
            tgt_components += 1
            tgt_misses += int(missed)
            target_mask |= component
        elif area >= 256:
            cmp_components += 1
            cmp_misses += int(missed)
    oracle = prediction.copy()
    oracle[target_mask] = target[target_mask]
    return {
        "baseline_miou": base,
        "target_oracle_gain": image_miou(target, oracle) - base,
        "target_components": tgt_components,
        "target_missed": tgt_misses,
        "comparator_components": cmp_components,
        "comparator_missed": cmp_misses,
    }


def _group_samples(rows: list[dict[str, Any]], folds: tuple[int, int], draws: int, seed: int):
    by_fold: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold[row["fold"]][row["specimen_group"]].append(row)
    for fold in folds:
        if len(by_fold[fold]) < 2:
            raise ValueError(f"fold {fold} requires >=2 specimen groups")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = []
        for fold in folds:
            groups = sorted(by_fold[fold])
            for index in rng.integers(0, len(groups), size=len(groups)):
                selected.extend(by_fold[fold][groups[int(index)]])
        samples.append(selected)
    return samples


def _miss_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    tc = sum(row["target_components"] for row in rows)
    tm = sum(row["target_missed"] for row in rows)
    cc = sum(row["comparator_components"] for row in rows)
    cm = sum(row["comparator_missed"] for row in rows)
    tr = tm / tc if tc else None
    cr = cm / cc if cc else None
    return {
        "target_components": float(tc),
        "target_miss_rate": tr,
        "comparator_components": float(cc),
        "comparator_miss_rate": cr,
        "target_minus_comparator_difference": tr - cr if tr is not None and cr is not None else None,
        "target_to_comparator_ratio": tr / cr if tr is not None and cr is not None and cr > 0 else None,
    }


def _ci(values: list[float]) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return np.percentile(finite, [2.5, 97.5]).astype(float).tolist() if finite.size else None


def headroom_audit(cv_root: Path, model382_root: Path, folds: tuple[int, int],
                   model382_lock: Path, model389_lock: Path, draws: int) -> dict[str, Any]:
    import model382_oof_ensemble_compare as model382

    if folds not in ((1, 2), (3, 4)):
        raise ValueError("run folds (1,2) exploratory or folds (3,4) primary")
    fold_masks, group_fold = load_fold_masks(cv_root)
    lock382 = json.loads(model382_lock.read_text(encoding="utf-8"))
    lock389 = _verify_model389_lock(model389_lock)
    for relative in model382.LOCKED_ARTIFACTS:
        path = model382.REPO_ROOT / relative
        if not path.is_file() or lock382.get("artifacts_sha256", {}).get(relative) != model382.sha256(path):
            raise ValueError(f"Model382 source lock mismatch: {relative}")
    seeds: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
    sources: dict[str, str] = {}
    for seed in (1, 2):
        rows = seeds[str(seed)]
        seen: set[str] = set()
        for fold in folds:
            masks = fold_masks[fold]
            oof_dir = model382_root / f"seed{seed}" / f"fold{fold}_oof"
            archive_path = model382._verify_model382_manifest(
                oof_dir, seed, fold, float(lock382["locked_at_unix"])
            )
            manifest_path = oof_dir / "model382_manifest.json"
            with np.load(archive_path, allow_pickle=False) as archive:
                if set(archive.files) != set(masks):
                    raise ValueError(f"OOF IDs differ from fixed train masks: {archive_path}")
                for image_id, mask_path in sorted(masks.items()):
                    if image_id in seen:
                        raise ValueError(f"duplicate OOF image: {image_id}")
                    seen.add(image_id)
                    target = read_mask(mask_path)
                    prob = np.asarray(archive[image_id], dtype=np.float32)
                    if prob.shape != (4, *target.shape) or not np.isfinite(prob).all():
                        raise ValueError(f"invalid probabilities for {image_id}")
                    if prob.min() < -1e-4 or prob.max() > 1.0001:
                        raise ValueError(f"probability range outside [0,1] for {image_id}")
                    if np.max(np.abs(prob.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized probabilities for {image_id}")
                    outcome = image_outcomes(target, prob.argmax(axis=0).astype(np.uint8))
                    rows.append({
                        "image_id": image_id,
                        "specimen_group": GROUP_RE.match(image_id).group(1),
                        "fold": fold,
                        **outcome,
                    })
                    sources[str(mask_path)] = sha256(mask_path)
            sources[str(archive_path)] = sha256(archive_path)
            sources[str(manifest_path)] = sha256(manifest_path)
            checkpoint = oof_dir.parent / f"fold{fold}" / "last.pt"
            sources[str(checkpoint)] = sha256(checkpoint)

    seed_results: dict[str, Any] = {}
    for seed, rows in seeds.items():
        samples = _group_samples(rows, folds, draws, BOOTSTRAP_SEED + int(seed))
        oracle_boot = [float(np.mean([row["target_oracle_gain"] for row in sample])) for sample in samples]
        miss_boot = [_miss_metrics(sample) for sample in samples]
        diff_boot = [float(x["target_minus_comparator_difference"])
                     for x in miss_boot if x["target_minus_comparator_difference"] is not None]
        class_result: dict[str, Any] = {
            "images": len(rows),
            "groups": len({row["specimen_group"] for row in rows}),
            "mean_imagewise_miou": float(np.mean([row["baseline_miou"] for row in rows])),
            "eutectic_64_to_lt256": {
                "target_components": sum(row["target_components"] for row in rows),
                "target_missed": sum(row["target_missed"] for row in rows),
                "comparator_components_ge256": sum(row["comparator_components"] for row in rows),
                "comparator_missed_ge256": sum(row["comparator_missed"] for row in rows),
                "oracle_mean_imagewise_miou_gain": float(np.mean([row["target_oracle_gain"] for row in rows])),
                "oracle_gain_ci95_group_bootstrap": _ci(oracle_boot),
                "component_miss_metrics": _miss_metrics(rows),
                "miss_difference_ci95_group_bootstrap": _ci(diff_boot),
                "per_fold": {},
            },
        }
        for fold in folds:
            fold_rows = [row for row in rows if row["fold"] == fold]
            class_result["eutectic_64_to_lt256"]["per_fold"][str(fold)] = {
                "images": len(fold_rows),
                "groups": len({row["specimen_group"] for row in fold_rows}),
                "oracle_mean_imagewise_miou_gain": float(np.mean([row["target_oracle_gain"] for row in fold_rows])),
                "component_miss_metrics": _miss_metrics(fold_rows),
            }
        seed_results[seed] = class_result
    return {
        "model": 389,
        "mode": "E2 held-out Model382 OOF component miss-rate and oracle headroom; not a sampler-effect estimate",
        "scope": "primary folds3-4" if folds == (3, 4) else "exploratory folds1-2",
        "folds": list(folds),
        "seed_results": seed_results,
        "component_miss_definition": "a GT Eutectic component is missed when fewer than 50% of its pixels are predicted Eutectic",
        "metric": "official per-image mean of four class IoUs; empty union=0",
        "bootstrap": {"draws": draws, "seed": BOOTSTRAP_SEED, "unit": "specimen group resampled within outer fold"},
        "model382_lock_sha256": sha256(model382_lock),
        "model389_lock_sha256": sha256(model389_lock),
        "source_sha256": sources,
        "official_valid_or_test_read": False,
        "training_started": False,
        "interpretation_guard": "OOF component headroom/miss rates are feasibility diagnostics, not causal sampler effects.",
    }


def _verify_model389_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in receipt["artifacts_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model389 lock hash mismatch: {relative}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exposure", "headroom"), required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs=2, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--model382-root", type=Path)
    parser.add_argument("--model382-lock", type=Path,
                        default=Path("experiments/results/model382_e5_preregistration_lock.json"))
    parser.add_argument("--model389-lock", type=Path,
                        default=Path("experiments/results/model389_preregistration_lock.json"))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = tuple(sorted(args.folds))
    if args.mode == "exposure":
        receipt = _verify_model389_lock(args.model389_lock)
        result = exposure_audit(args.cv_root, folds)
        result["model389_lock_sha256"] = sha256(args.model389_lock)
        result["model389_locked_at_utc"] = receipt.get("locked_at_utc")
    else:
        if args.model382_root is None:
            parser.error("--model382-root is required for headroom mode")
        if args.bootstrap_draws < 100:
            parser.error("--bootstrap-draws must be >=100")
        result = headroom_audit(args.cv_root, args.model382_root, folds,
                                args.model382_lock, args.model389_lock, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_sha256", "seed_results")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
