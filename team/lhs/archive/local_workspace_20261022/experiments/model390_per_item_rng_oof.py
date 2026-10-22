#!/usr/bin/env python3
"""Locked, paired train-OOF analysis for Model390 per-item RNG correction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from model382_oof_ensemble_compare import (
    GROUP_RE,
    LOCKED_ARTIFACTS,
    STEP_RE,
    _load_archive,
    _paired_bootstrap,
    _verify_model382_manifest,
    class_ious,
    mask_to_ids,
)


CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 390
METRIC = "image-wise 4-class mIoU; empty-union=0"
INFERENCE = {
    "patch_size": 512,
    "overlap": 170,
    "input_scale": 2.0,
    "tta": "flip4",
    "downsample_antialias": False,
    "batch_size": 2,
    "device": "cuda:0",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model390_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != "Model390 per-item crop and augmentation RNG correction":
        raise ValueError("unexpected Model390 lock identity")
    locked_files = {
        receipt["protocol"]["path"]: receipt["protocol"]["sha256"],
        **receipt["candidate_source_sha256"],
        receipt["synthetic_smoke"]["path"]: receipt["synthetic_smoke"]["sha256"],
    }
    runner = receipt["reference_oof_runner"]
    runner_name = Path(runner["path"]).name
    if runner_name != "run_model382_seed_count_control_p100.sh":
        raise ValueError("unexpected locked Model382 reference runner")
    locked_files[runner_name] = runner["sha256"]
    analysis_hash = receipt.get("analysis_source_sha256")
    if not analysis_hash:
        raise ValueError("Model390 OOF evaluator is not included in the lock receipt")
    locked_files.update(analysis_hash)
    smoke_hash = receipt.get("analysis_smoke_sha256")
    if not smoke_hash:
        raise ValueError("Model390 OOF smoke is not included in the lock receipt")
    locked_files.update(smoke_hash)
    dependency_hash = receipt.get("analysis_dependency_sha256")
    if not dependency_hash:
        raise ValueError("Model390 input-verification dependencies are not included in the lock receipt")
    locked_files.update(dependency_hash)
    for relative, expected in locked_files.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model390 lock/source hash mismatch: {relative}")
    return receipt


def expected_training_args(root: Path, run_dir: Path, seed: int) -> dict[str, Any]:
    return {
        "data_root": str(root / f"data_model382_seed{seed}_fold{run_dir.name.removeprefix('fold')}"),
        "output_dir": str(run_dir),
        "device": "cuda:0",
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet",
        "patch_size": 512,
        "input_scale": 2.0,
        "samples_per_image": 12,
        "batch_size": 4,
        "accumulation_steps": 2,
        "num_workers": 4,
        "epochs": 70,
        "lr": 3e-4,
        "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4,
        "no_boundary_head": False,
        "boundary_head_weight": 1.0,
        "ce_class_weights": None,
        "init_checkpoint": None,
        "seed": seed,
        "rng_mode": "per_item",
        "val_every": 0,
        "no_amp": False,
        "resume": True,
    }


def verify_candidate_manifest(
    root: Path,
    run_dir: Path,
    oof_dir: Path,
    seed: int,
    fold: int,
    model390_lock: Path,
    lock_sha256: str,
) -> Path:
    archive = oof_dir / "valid_probabilities.npz"
    manifest_path = oof_dir / "model390_manifest.json"
    checkpoint = run_dir / "last.pt"
    args_path = run_dir / "args.json"
    if not all(path.is_file() for path in (archive, manifest_path, checkpoint, args_path)):
        raise FileNotFoundError(f"missing Model390 checkpoint/args/OOF manifest: {oof_dir}")
    locked_at = lock_timestamp(model390_lock)
    if any(path.stat().st_mtime <= locked_at for path in (archive, manifest_path, checkpoint, args_path)):
        raise ValueError(f"Model390 artifact predates its pre-score lock: {oof_dir}")

    expected_args = expected_training_args(root, run_dir, seed)
    actual_args = json.loads(args_path.read_text(encoding="utf-8"))
    if actual_args != expected_args:
        different = {
            key: (actual_args.get(key), value)
            for key, value in expected_args.items()
            if actual_args.get(key) != value
        }
        raise ValueError(f"Model390 training args mismatch in {args_path}: {different}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "model": 390,
        "seed": seed,
        "fold": fold,
        "epochs": 70,
        "rng_mode": "per_item",
        "metric": METRIC,
        "checkpoint_sha256": sha256(checkpoint),
        "oof_probabilities_sha256": sha256(archive),
        "model390_lock_sha256": lock_sha256,
        "inference": INFERENCE,
        "training_args": expected_args,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Model390 manifest mismatch {manifest_path}: {key}")
    return archive


def lock_timestamp(lock_path: Path) -> float:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    value = receipt["locked_at_utc"]
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Model390 lock timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def _score_pair(reference_probs: np.ndarray, candidate_probs: np.ndarray, truth: np.ndarray):
    reference_iou = class_ious(reference_probs, truth)
    candidate_iou = class_ious(candidate_probs, truth)
    return reference_iou, candidate_iou


def load_paired_rows(
    cv_root: Path,
    model382_root: Path,
    model390_root: Path,
    model382_lock: Path,
    model390_lock: Path,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, dict[str, Any]]]:
    receipt = verify_model390_lock(model390_lock)
    locked_model382_lock = REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json"
    if model382_lock.resolve() != locked_model382_lock.resolve():
        raise ValueError("Model390 is bound to the fixed Model382 preregistration receipt")
    if sha256(model382_lock) != receipt["analysis_dependency_sha256"][str(
        locked_model382_lock.relative_to(REPO_ROOT)
    )]:
        raise ValueError("Model382 preregistration receipt hash differs from the Model390 lock")
    lock_sha = sha256(model390_lock)
    primary_receipt = json.loads(model382_lock.read_text(encoding="utf-8"))
    primary_hashes = primary_receipt.get("artifacts_sha256", {})
    for relative in LOCKED_ARTIFACTS:
        path = REPO_ROOT / relative
        if not path.is_file() or primary_hashes.get(relative) != sha256(path):
            raise ValueError(f"Model382 pre-result source lock mismatch: {relative}")
    locked_at = float(primary_receipt["locked_at_unix"])
    rows = {seed: {} for seed in seeds}
    seen: set[str] = set()
    group_fold: dict[str, int] = {}

    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        masks = {path.name.removesuffix("_mask.png"): path
                 for path in mask_dir.glob("*_mask.png")}
        if not masks:
            raise FileNotFoundError(f"missing held-out train masks: {mask_dir}")
        expected_ids = set(masks)

        for seed in seeds:
            reference_dir = model382_root / f"seed{seed}/fold{fold}_oof"
            reference_archive = _verify_model382_manifest(
                reference_dir, seed, fold, locked_at
            )
            candidate_run = model390_root / f"seed{seed}/fold{fold}"
            candidate_oof = model390_root / f"seed{seed}/fold{fold}_oof"
            candidate_archive = verify_candidate_manifest(
                model390_root.parents[1],
                candidate_run,
                candidate_oof,
                seed,
                fold,
                model390_lock,
                lock_sha,
            )

            reference = _load_archive(reference_archive, expected_ids)
            candidate = _load_archive(candidate_archive, expected_ids)
            try:
                for image_id in sorted(expected_ids):
                    if image_id in seen and seed == seeds[0]:
                        raise ValueError(f"duplicate OOF image across folds: {image_id}")
                    group_match = GROUP_RE.match(image_id)
                    step_match = STEP_RE.search(image_id)
                    if not group_match or not step_match:
                        raise ValueError(f"cannot parse specimen/step from {image_id}")
                    group = group_match.group(1)
                    previous = group_fold.setdefault(group, fold)
                    if previous != fold:
                        raise ValueError(f"specimen group crosses folds: {group}")
                    truth = mask_to_ids(masks[image_id])
                    ref_iou, cand_iou = _score_pair(
                        np.asarray(reference[image_id], dtype=np.float32),
                        np.asarray(candidate[image_id], dtype=np.float32),
                        truth,
                    )
                    if seed == seeds[0]:
                        seen.add(image_id)
                    rows[seed][image_id] = {
                        "group": group,
                        "fold": fold,
                        "step": int(step_match.group(1)),
                        "reference_class_iou": ref_iou,
                        "candidate_class_iou": cand_iou,
                    }
            finally:
                reference.close()
                candidate.close()

    groups = {row["group"] for row in rows[seeds[0]].values()}
    if len(seen) != 70 or len(groups) != 59:
        raise ValueError(f"expected 70 images/59 groups; got {len(seen)}/{len(groups)}")
    for seed in seeds:
        if set(rows[seed]) != seen:
            raise ValueError(f"seed{seed} OOF coverage differs from the fixed 70-image set")
    return rows


def summarize_seed(rows: dict[str, dict[str, Any]], draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    names = sorted(rows)
    if len(names) != 70:
        raise ValueError(f"expected 70 paired images, got {len(names)}")
    reference = np.asarray([rows[name]["reference_class_iou"] for name in names])
    candidate = np.asarray([rows[name]["candidate_class_iou"] for name in names])
    image_delta = candidate.mean(axis=1) - reference.mean(axis=1)
    groups = [rows[name]["group"] for name in names]
    folds = [int(rows[name]["fold"]) for name in names]
    steps = np.asarray([int(rows[name]["step"]) for name in names])
    step4 = steps == 4
    if not step4.any():
        raise ValueError("Step4 subset is empty")
    class_delta = (candidate - reference).mean(axis=0) * 100.0
    fold_delta = {
        str(fold): float(image_delta[np.asarray(folds) == fold].mean() * 100.0)
        for fold in range(1, 5)
    }
    ci = _paired_bootstrap(image_delta, groups, folds, draws, BOOTSTRAP_SEED)
    delta_pp = float(image_delta.mean() * 100.0)
    step4_delta = float(image_delta[step4].mean() * 100.0)
    gate = {
        "gain_at_least_0_50pp": delta_pp >= 0.50,
        "group_bootstrap_ci_lower_above_zero": ci[0] > 0.0,
        "at_least_3_of_4_folds_positive": sum(value > 0 for value in fold_delta.values()) >= 3,
        "worst_fold_at_least_minus_0_30pp": min(fold_delta.values()) >= -0.30,
        "all_class_deltas_at_least_minus_0_50pp": bool(np.all(class_delta >= -0.50)),
        "step4_delta_at_least_minus_0_50pp": step4_delta >= -0.50,
    }
    return {
        "reference_image_miou_empty0_pct": float(reference.mean(axis=1).mean() * 100.0),
        "candidate_image_miou_empty0_pct": float(candidate.mean(axis=1).mean() * 100.0),
        "delta_pp": delta_pp,
        "group_bootstrap_95ci_delta_pp": ci,
        "class_delta_pp": {name: float(value) for name, value in zip(CLASS_NAMES, class_delta)},
        "fold_delta_pp": fold_delta,
        "step4_delta_pp": step4_delta,
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def summarize_final(rows: dict[int, dict[str, dict[str, Any]]],
                    draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if set(rows) != {1, 2}:
        raise ValueError("final stage requires paired OOF for seeds 1 and 2")
    per_seed = {seed: summarize_seed(rows[seed], draws) for seed in (1, 2)}
    names = sorted(rows[1])
    for name in names:
        for field in ("group", "fold", "step"):
            if rows[1][name][field] != rows[2][name][field]:
                raise ValueError(f"seed metadata mismatch for {name}: {field}")

    ref_by_seed = np.asarray([
        [rows[seed][name]["reference_class_iou"] for name in names]
        for seed in (1, 2)
    ])
    cand_by_seed = np.asarray([
        [rows[seed][name]["candidate_class_iou"] for name in names]
        for seed in (1, 2)
    ])
    delta_by_seed_image = cand_by_seed.mean(axis=2) - ref_by_seed.mean(axis=2)
    pooled_delta = delta_by_seed_image.mean(axis=0)
    groups = [rows[1][name]["group"] for name in names]
    folds = [int(rows[1][name]["fold"]) for name in names]
    boot_delta = delta_by_seed_image.reshape(-1)
    ci = _paired_bootstrap(
        boot_delta,
        groups * 2,
        folds * 2,
        draws,
        BOOTSTRAP_SEED,
    )
    pooled_class_delta = (cand_by_seed - ref_by_seed).mean(axis=(0, 1)) * 100.0
    pooled_fold_delta = {
        str(fold): float(pooled_delta[np.asarray(folds) == fold].mean() * 100.0)
        for fold in range(1, 5)
    }
    steps = np.asarray([rows[1][name]["step"] for name in names])
    step4 = steps == 4
    pooled_step4_delta = float(pooled_delta[step4].mean() * 100.0)
    per_seed_safety = all(
        summary["delta_pp"] > 0.0
        and sum(value > 0 for value in summary["fold_delta_pp"].values()) >= 3
        and min(summary["fold_delta_pp"].values()) >= -0.30
        and all(value >= -0.50 for value in summary["class_delta_pp"].values())
        and summary["step4_delta_pp"] >= -0.50
        for summary in per_seed.values()
    )
    pooled_safety = (
        min(pooled_fold_delta.values()) >= -0.30
        and bool(np.all(pooled_class_delta >= -0.50))
        and pooled_step4_delta >= -0.50
    )
    gate = {
        "both_seed_point_deltas_positive": all(per_seed[s]["delta_pp"] > 0 for s in (1, 2)),
        "equal_weight_mean_gain_at_least_0_50pp": float(pooled_delta.mean() * 100.0) >= 0.50,
        "paired_group_bootstrap_ci_lower_above_zero": ci[0] > 0.0,
        "each_seed_fold_class_step4_safety_passed": per_seed_safety,
        "pooled_fold_class_step4_safety_passed": pooled_safety,
    }
    return {
        "per_seed": per_seed,
        "equal_weight_delta_pp": float(pooled_delta.mean() * 100.0),
        "equal_weight_group_bootstrap_95ci_delta_pp": ci,
        "equal_weight_class_delta_pp": {
            name: float(value) for name, value in zip(CLASS_NAMES, pooled_class_delta)
        },
        "equal_weight_fold_delta_pp": pooled_fold_delta,
        "equal_weight_step4_delta_pp": pooled_step4_delta,
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_model390_lock(args.model390_lock)
    lock_sha = sha256(args.model390_lock)
    if args.stage == "seed1":
        seeds = (1,)
    else:
        if args.seed1_result is None or not args.seed1_result.is_file():
            raise ValueError("final stage requires the seed1 result JSON")
        first = json.loads(args.seed1_result.read_text(encoding="utf-8"))
        if (first.get("stage") != "seed1" or not first.get("continuation_gate_passed")
                or first.get("model390_lock_sha256") != lock_sha):
            raise ValueError("seed1 continuation gate/lock does not authorize final stage")
        seeds = (1, 2)

    rows = load_paired_rows(
        args.cv_root,
        args.model382_root,
        args.model390_root,
        args.model382_lock,
        args.model390_lock,
        seeds,
    )
    if args.stage == "seed1":
        summary = summarize_seed(rows[1])
        result = {
            "model": 390,
            "stage": "seed1",
            "scope": "70 train-only grouped OOF images / 59 specimen groups",
            "metric": METRIC,
            "model390_lock_sha256": lock_sha,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "summary": summary,
            "continuation_gate_passed": summary["gate_passed"],
            "valid_evaluation_authorized": False,
        }
    else:
        summary = summarize_final(rows)
        result = {
            "model": 390,
            "stage": "final",
            "scope": "70 train-only grouped OOF images / 59 specimen groups, paired across seeds",
            "metric": METRIC,
            "model390_lock_sha256": lock_sha,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "summary": summary,
            "valid_evaluation_authorized": summary["gate_passed"],
        }
    result["sources"] = {
        "model390_lock": str(args.model390_lock),
        "model382_lock": str(args.model382_lock),
        "cv_root": str(args.cv_root),
        "model382_root": str(args.model382_root),
        "model390_root": str(args.model390_root),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior analysis output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": result["stage"], "summary": summary,
                      "valid_evaluation_authorized": result["valid_evaluation_authorized"]},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("seed1", "final"), required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--model390-root", type=Path, required=True)
    parser.add_argument("--model382-lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json")
    parser.add_argument("--model390-lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model390_preregistration_lock.json")
    parser.add_argument("--seed1-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
