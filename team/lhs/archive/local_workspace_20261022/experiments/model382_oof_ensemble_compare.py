#!/usr/bin/env python3
"""Evaluate the fixed Model382 seed-count arms and pre-registered E5 OOF arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
ARMS = ("A", "S1", "S2", "B", "C", "E5")
LOCKED_ARTIFACTS = (
    "experiments/model382_seed_count_control_preregistration.md",
    "experiments/model382_five_model_ensemble_addendum.md",
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/model382_oof_ensemble_compare_smoke.py",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=2)] = class_id
    if np.any(result == 255):
        raise ValueError(f"unknown mask RGB values in {path}")
    return result


def class_ious(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values {probabilities.shape} for {truth.shape}")
    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError("probability values outside [0,1]")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError("probability map is not normalized")
    prediction = probabilities.argmax(axis=0)
    matrix = np.bincount(
        truth.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def _verify_model382_manifest(oof_dir: Path, seed: int, fold: int,
                              locked_at_unix: float) -> Path:
    archive_path = oof_dir / "valid_probabilities.npz"
    manifest_path = oof_dir / "model382_manifest.json"
    if not archive_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing Model382 archive/manifest: {oof_dir}")
    if archive_path.stat().st_mtime <= locked_at_unix or manifest_path.stat().st_mtime <= locked_at_unix:
        raise ValueError(f"Model382 OOF artifact predates or matches preregistration lock: {oof_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": 382,
        "seed": seed,
        "fold": fold,
        "epochs": 70,
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "oof_probabilities_sha256": sha256(archive_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Model382 manifest mismatch {manifest_path}: {key}")
    checkpoint = oof_dir.parent / f"fold{fold}" / "last.pt"
    if not checkpoint.is_file() or manifest.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"Model382 checkpoint missing or hash mismatch: {checkpoint}")
    args = manifest.get("training_args", {})
    expected_args = {
        "seed": seed,
        "epochs": 70,
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet",
        "patch_size": 512,
        "input_scale": 2.0,
        "samples_per_image": 12,
        "batch_size": 4,
        "accumulation_steps": 2,
        "lr": 3e-4,
        "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4,
        "boundary_head_weight": 1.0,
        "val_every": 0,
        "init_checkpoint": None,
    }
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError(f"Model382 training arguments mismatch in {manifest_path}")
    return archive_path


def _load_archive(path: Path, expected_ids: set[str]):
    archive = np.load(path, allow_pickle=False)
    if set(archive.files) != expected_ids:
        archive.close()
        raise ValueError(f"OOF IDs differ from masks in {path}")
    return archive


def load_oof(cv_root: Path, model374_root: Path, model376_root: Path,
             model377_root: Path, model382_root: Path,
             locked_at_unix: float) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    seen: set[str] = set()
    group_folds: dict[str, int] = {}

    for fold in range(1, 5):
        split_dir = cv_root / f"oof_fold{fold}"
        mask_dir = split_dir / "masks"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"missing fixed OOF masks: {mask_dir}")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if not masks:
            raise ValueError(f"no OOF masks in {mask_dir}")
        expected_ids = set(masks)

        paths = {
            "A": model374_root / f"fold{fold}_oof/valid_probabilities.npz",
            "C374": model374_root / f"fold{fold}_oof/valid_probabilities.npz",
            "C376": model376_root / f"fold{fold}_oof/valid_probabilities.npz",
            "C377": model377_root / f"fold{fold}_oof/valid_probabilities.npz",
            "S1": _verify_model382_manifest(model382_root / f"seed1/fold{fold}_oof", 1, fold, locked_at_unix),
            "S2": _verify_model382_manifest(model382_root / f"seed2/fold{fold}_oof", 2, fold, locked_at_unix),
        }
        paths["A"] = paths["C374"]
        archives = {name: _load_archive(path, expected_ids) for name, path in paths.items()}
        try:
            for image_id in sorted(expected_ids):
                if image_id in seen:
                    raise ValueError(f"image appears in multiple OOF folds: {image_id}")
                seen.add(image_id)
                group_match = GROUP_RE.match(image_id)
                step_match = STEP_RE.search(image_id)
                if not group_match or not step_match:
                    raise ValueError(f"cannot parse group/step from {image_id}")
                group = group_match.group(1)
                previous_fold = group_folds.setdefault(group, fold)
                if previous_fold != fold:
                    raise ValueError(f"specimen group {group} appears in folds {previous_fold} and {fold}")
                truth = mask_to_ids(masks[image_id])
                probability = {
                    name: np.asarray(archive[image_id], dtype=np.float32)
                    for name, archive in archives.items()
                }
                reference_shape = (4, *truth.shape)
                for name, value in probability.items():
                    if value.shape != reference_shape:
                        raise ValueError(f"{name} probability shape mismatch for {image_id}")
                    # class_ious validates finiteness, range and per-pixel normalization.
                    class_ious(value, truth)

                mixtures = {
                    "A": probability["A"],
                    "S1": probability["S1"],
                    "S2": probability["S2"],
                    "B": (probability["A"] + probability["S1"] + probability["S2"]) / 3.0,
                    "C": (probability["C374"] + probability["C376"] + probability["C377"]) / 3.0,
                    "E5": (probability["A"] + probability["C376"] + probability["C377"]
                           + probability["S1"] + probability["S2"]) / 5.0,
                }
                for arm, mixture in mixtures.items():
                    rows[arm][image_id] = {
                        "fold": fold,
                        "group": group,
                        "step": int(step_match.group(1)),
                        "class_iou": class_ious(mixture, truth),
                    }
        finally:
            for archive in archives.values():
                archive.close()

    groups = {row["group"] for row in rows["A"].values()}
    if len(seen) != 70 or len(groups) != 59:
        raise ValueError(f"expected exactly 70 images/59 groups; got {len(seen)}/{len(groups)}")
    for arm in ARMS:
        if set(rows[arm]) != seen:
            raise ValueError(f"{arm} OOF coverage differs from reference")
    return rows


def _paired_bootstrap(delta: np.ndarray, groups: list[str], folds: list[int],
                      draws: int, seed: int) -> list[float]:
    grouped: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    groups_array = np.asarray(groups)
    folds_array = np.asarray(folds)
    for fold in range(1, 5):
        for group in sorted(set(groups_array[folds_array == fold])):
            grouped[fold][group] = np.flatnonzero((folds_array == fold) & (groups_array == group))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected: list[np.ndarray] = []
        for fold in range(1, 5):
            names = sorted(grouped[fold])
            sampled = rng.integers(0, len(names), size=len(names))
            selected.extend(grouped[fold][names[int(i)]] for i in sampled)
        indices = np.concatenate(selected)
        boot[draw] = delta[indices].mean()
    return np.percentile(boot * 100.0, [2.5, 97.5]).astype(float).tolist()


def paired_comparison(reference: dict[str, dict], candidate: dict[str, dict],
                      draws: int, seed: int) -> dict[str, Any]:
    names = sorted(reference)
    if names != sorted(candidate) or len(names) != 70:
        raise ValueError("paired comparison requires identical coverage of 70 OOF images")
    groups = [reference[name]["group"] for name in names]
    folds = [int(reference[name]["fold"]) for name in names]
    deltas = np.asarray([
        np.mean(candidate[name]["class_iou"]) - np.mean(reference[name]["class_iou"])
        for name in names
    ], dtype=np.float64)
    class_delta = np.mean(np.asarray([
        candidate[name]["class_iou"] - reference[name]["class_iou"] for name in names
    ]), axis=0) * 100.0
    fold_delta = {
        str(fold): float(deltas[np.asarray(folds) == fold].mean() * 100.0)
        for fold in range(1, 5)
    }
    group_delta = {
        group: float(deltas[np.asarray(groups) == group].mean() * 100.0)
        for group in sorted(set(groups))
    }
    a71 = np.asarray([group == "A71" for group in groups])
    step4 = np.asarray([int(reference[name]["step"]) == 4 for name in names])
    if not a71.any() or not step4.any():
        raise ValueError("A71 and Step4 sensitivity subsets must be present")
    return {
        "reference_image_mean_miou_empty0_pct": float(np.mean([
            np.mean(reference[name]["class_iou"]) for name in names
        ]) * 100.0),
        "candidate_image_mean_miou_empty0_pct": float(np.mean([
            np.mean(candidate[name]["class_iou"]) for name in names
        ]) * 100.0),
        "delta_pp": float(deltas.mean() * 100.0),
        "paired_fold_stratified_group_bootstrap_95ci_delta_pp":
            _paired_bootstrap(deltas, groups, folds, draws, seed),
        "class_mean_iou_delta_pp": {
            CLASS_NAMES[i]: float(class_delta[i]) for i in range(4)
        },
        "fold_delta_pp": fold_delta,
        "positive_images": int((deltas > 0).sum()),
        "positive_specimen_groups": int(sum(value > 0 for value in group_delta.values())),
        "per_specimen_group_delta_pp": group_delta,
        "A71_delta_pp": float(deltas[a71].mean() * 100.0),
        "A71_excluded_delta_pp": float(deltas[~a71].mean() * 100.0),
        "step4_images": int(step4.sum()),
        "step4_delta_pp": float(deltas[step4].mean() * 100.0),
    }


def summarize_arm(rows: dict[str, dict], draws: int, seed: int) -> dict[str, Any]:
    names = sorted(rows)
    scores = np.asarray([np.mean(rows[name]["class_iou"]) for name in names], dtype=np.float64)
    groups = [rows[name]["group"] for name in names]
    folds = [int(rows[name]["fold"]) for name in names]
    return {
        "images": len(names),
        "specimen_groups": len(set(groups)),
        "image_mean_miou_empty0_pct": float(scores.mean() * 100.0),
        "paired_fold_stratified_group_bootstrap_95ci_pct": _paired_bootstrap(
            scores, groups, folds, draws, seed
        ),
        "class_mean_iou_pct": {
            CLASS_NAMES[i]: float(np.mean([rows[name]["class_iou"][i] for name in names]) * 100.0)
            for i in range(4)
        },
        "fold_image_mean_miou_pct": {
            str(fold): float(np.mean([scores[i] for i, name in enumerate(names)
                                      if rows[name]["fold"] == fold]) * 100.0)
            for fold in range(1, 5)
        },
    }


def _absolute_gate(comparison: dict[str, Any]) -> dict[str, bool]:
    return {
        "gain_at_least_0_5pp": comparison["delta_pp"] >= 0.5,
        "group_bootstrap_ci_lower_above_zero":
            comparison["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] > 0,
        "at_least_3_of_4_folds_positive":
            sum(value > 0 for value in comparison["fold_delta_pp"].values()) >= 3,
        "worst_fold_at_least_minus_0_3pp": min(comparison["fold_delta_pp"].values()) >= -0.3,
        "no_class_mean_iou_drop_over_0_5pp":
            all(value >= -0.5 for value in comparison["class_mean_iou_delta_pp"].values()),
        "step4_delta_at_least_minus_0_5pp": comparison["step4_delta_pp"] >= -0.5,
    }


def run(cv_root: Path, model374_root: Path, model376_root: Path,
        model377_root: Path, model382_root: Path, prereg_lock: Path,
        draws: int = 20_000) -> dict[str, Any]:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    lock = json.loads(prereg_lock.read_text(encoding="utf-8"))
    locked_at_unix = float(lock["locked_at_unix"])
    artifact_hashes = lock.get("artifacts_sha256", {})
    for relative in LOCKED_ARTIFACTS:
        path = REPO_ROOT / relative
        if not path.is_file() or artifact_hashes.get(relative) != sha256(path):
            raise ValueError(f"pre-result lock hash mismatch: {relative}")
    rows = load_oof(cv_root, model374_root, model376_root, model377_root,
                    model382_root, locked_at_unix)
    scores = {arm: summarize_arm(rows[arm], draws, 382 + index)
              for index, arm in enumerate(ARMS)}
    comparisons = {
        "S1_vs_A": paired_comparison(rows["A"], rows["S1"], draws, 382),
        "S2_vs_A": paired_comparison(rows["A"], rows["S2"], draws, 382),
        "B_vs_A": paired_comparison(rows["A"], rows["B"], draws, 382),
        "C_vs_B": paired_comparison(rows["B"], rows["C"], draws, 382),
        "C_vs_A": paired_comparison(rows["A"], rows["C"], draws, 382),
        "E5_vs_A": paired_comparison(rows["A"], rows["E5"], draws, 382),
        "E5_vs_B": paired_comparison(rows["B"], rows["E5"], draws, 382),
        "E5_vs_C": paired_comparison(rows["C"], rows["E5"], draws, 382),
    }
    diversity = comparisons["C_vs_B"]
    seed_count = comparisons["B_vs_A"]
    diversity_signal = (
        diversity["delta_pp"] >= 0.20
        and diversity["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] > 0
        and sum(value > 0 for value in diversity["fold_delta_pp"].values()) >= 3
    )
    count_signal = (
        not diversity_signal
        and diversity["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] <= 0
        <= diversity["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][1]
        and seed_count["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] > 0
    )
    b_relative_superiority = (
        diversity["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][1] < 0
    )
    primary_candidate = (
        "C" if diversity_signal else "B" if count_signal or b_relative_superiority else None
    )
    primary_gate = (_absolute_gate(comparisons[f"{primary_candidate}_vs_A"])
                    if primary_candidate else None)

    e5_vs_b = comparisons["E5_vs_B"]
    e5_vs_c = comparisons["E5_vs_C"]
    e5_gate = {
        "E5_vs_A_passes_existing_absolute_gate": all(_absolute_gate(comparisons["E5_vs_A"]).values()),
        "E5_vs_A_A71_excluded_nonnegative": comparisons["E5_vs_A"]["A71_excluded_delta_pp"] >= 0,
        "E5_vs_B_gain_at_least_0_2pp_and_ci_lower_above_zero": (
            e5_vs_b["delta_pp"] >= 0.20
            and e5_vs_b["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] > 0
        ),
        "E5_vs_C_gain_at_least_0_2pp_and_ci_lower_above_zero": (
            e5_vs_c["delta_pp"] >= 0.20
            and e5_vs_c["paired_fold_stratified_group_bootstrap_95ci_delta_pp"][0] > 0
        ),
        "at_least_3_of_4_folds_positive_vs_B_and_C": (
            sum(value > 0 for value in e5_vs_b["fold_delta_pp"].values()) >= 3
            and sum(value > 0 for value in e5_vs_c["fold_delta_pp"].values()) >= 3
        ),
        "A71_excluded_delta_nonnegative_vs_B_and_C": (
            e5_vs_b["A71_excluded_delta_pp"] >= 0
            and e5_vs_c["A71_excluded_delta_pp"] >= 0
        ),
    }
    standalone_scores = np.asarray([
        scores[arm]["image_mean_miou_empty0_pct"] for arm in ("A", "S1", "S2")
    ])
    return {
        "model": 382,
        "scope": "70 train images, exactly-once specimen-group OOF; no official valid/test inputs or labels",
        "metric": "per-image arithmetic mean of four class IoUs; empty-union=0",
        "bootstrap": {
            "unit": "specimen group resampled with replacement within each fixed outer fold; all images in sampled group carried",
            "draws": draws,
            "seed": 382,
        },
        "pre_result_lock": {
            "locked_at_utc": lock["locked_at_utc"],
            "locked_at_unix": locked_at_unix,
            "lock_manifest_sha256": sha256(prereg_lock),
            "artifact_hashes_verified": list(LOCKED_ARTIFACTS),
            "OOF_archives_and_manifests_required_newer_than_lock": True,
        },
        "arms": {
            "A": "Model374 seed0",
            "S1": "Model382 seed1",
            "S2": "Model382 seed2",
            "B": "uniform probability mean of Model374 seed0 + Model382 seed1 + Model382 seed2",
            "C": "uniform probability mean of Model374 + Model376 + Model377 (Model379)",
            "E5": "fixed exploratory uniform probability mean of all five models; preregistered secondary arm",
        },
        "model_scores": scores,
        "comparisons": comparisons,
        "seed_noise_descriptive_only_n3": {
            "standalone_arm_means_pct": {arm: float(scores[arm]["image_mean_miou_empty0_pct"])
                                          for arm in ("A", "S1", "S2")},
            "standard_deviation_pp": float(standalone_scores.std(ddof=1)),
            "not_a_seed_variance_confidence_interval": True,
        },
        "primary_decision": {
            "diversity_signal_C_vs_B": diversity_signal,
            "model_count_signal_B_vs_A": count_signal,
            "B_relative_superiority_CI_upper_below_zero": b_relative_superiority,
            "candidate_if_signal": primary_candidate,
            "candidate_absolute_adoption_gate": primary_gate,
            "candidate_adopted": bool(primary_gate is not None and all(primary_gate.values())),
        },
        "E5_secondary_gate": e5_gate,
        "E5_secondary_gate_passed": all(e5_gate.values()),
        "interpretation_guard": (
            "Primary A/B/C conclusions follow the unchanged Model382 preregistration. E5 cannot alter those conclusions. "
            "All results are selection-conditional train OOF evidence, not independent test performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--model376-root", type=Path, required=True)
    parser.add_argument("--model377-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--pre-result-lock", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(args.cv_root, args.model374_root, args.model376_root,
                 args.model377_root, args.model382_root, args.pre_result_lock, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": result["model"],
        "model_scores": {key: value["image_mean_miou_empty0_pct"]
                         for key, value in result["model_scores"].items()},
        "primary_decision": result["primary_decision"],
        "E5_gate": result["E5_secondary_gate"],
        "E5_gate_passed": result["E5_secondary_gate_passed"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
