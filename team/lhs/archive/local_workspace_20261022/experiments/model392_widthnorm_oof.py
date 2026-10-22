#!/usr/bin/env python3
"""Locked paired train-OOF analysis for Model392 width normalization."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from model382_oof_ensemble_compare import (
    GROUP_RE,
    LOCKED_ARTIFACTS,
    STEP_RE,
    _load_archive,
    _verify_model382_manifest,
    class_ious,
    mask_to_ids,
)
from model390_per_item_rng_oof import (
    CLASS_NAMES,
    METRIC,
    summarize_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 390
CONFIRMATORY_OUTPUT = REPO_ROOT / "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
BASE_INFERENCE = {
    "patch_size": 512,
    "base_input_scale": 2.0,
    "overlap_base": 170,
    "overlap_rule": "preserve overlap/native-tile ratio",
    "tta": "flip4",
    "tta_mode": "flip4",
    "tta_enabled": True,
    "downsample_antialias": False,
    "batch_size": 2,
    "device": "cuda:0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def claim_confirmatory_analysis(output: Path, lock_sha256: str) -> tuple[Path, str]:
    output = output.resolve()
    if output != CONFIRMATORY_OUTPUT.resolve():
        raise ValueError(f"confirmatory OOF analysis has a single fixed output path: {CONFIRMATORY_OUTPUT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior OOF gate result: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse a leftover OOF result temporary: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    claim = output.with_suffix(output.suffix + ".claim")
    record = {
        "model": 392,
        "stage": "two_seed_confirmatory_claimed",
        "model392_lock_sha256": lock_sha256,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": "exclusive one-shot analysis claim; retained after interruption",
    }
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"confirmatory OOF analysis was already claimed: {claim}") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return claim, sha256(claim)


def verify_model392_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != "Model392 per-image pixel-scale normalization":
        raise ValueError("unexpected Model392 lock identity")
    if receipt.get("revision") in (2, 3):
        previous = receipt.get("supersedes", {})
        previous_path = REPO_ROOT / previous.get("path", "")
        if not previous_path.is_file() or sha256(previous_path) != previous.get("sha256"):
            raise ValueError("Model392 amended lock does not match its superseded lock")
        locked_files = receipt.get("locked_files_sha256")
        if not locked_files or locked_files.get(previous.get("path")) != previous.get("sha256"):
            raise ValueError("Model392 amended lock omits its superseded receipt from the hash manifest")

        categorized: dict[str, str] = {}
        protocol = receipt.get("protocol", {})
        if protocol.get("path") and protocol.get("sha256"):
            categorized[protocol["path"]] = protocol["sha256"]
        categorized.update(protocol.get("supersedes_original", {}))
        for section in (
            "candidate_source_sha256",
            "analysis_source_sha256",
            "analysis_smoke_sha256",
            "execution_source_sha256",
            "analysis_dependency_sha256",
        ):
            categorized.update(receipt.get(section, {}))
        geometry = receipt.get("geometry_audit", {})
        if geometry.get("path") and geometry.get("sha256"):
            categorized[geometry["path"]] = geometry["sha256"]
        generator = receipt.get("lock_generator", {})
        if generator.get("path") and generator.get("sha256"):
            categorized[generator["path"]] = generator["sha256"]
        runner = receipt.get("reference_oof_runner", {})
        runner_path = runner.get("local_path")
        runner_hash = runner.get("sha256")
        if runner_path and runner_hash:
            categorized[runner_path] = runner_hash
        categorized[previous["path"]] = previous["sha256"]
        for relative, expected in categorized.items():
            if locked_files.get(relative) != expected:
                raise ValueError(f"Model392 lock metadata/hash-map mismatch: {relative}")
        if ("experiments/results/model382_e5_preregistration_lock.json"
                not in receipt.get("analysis_dependency_sha256", {})):
            raise ValueError("Model392 amended lock omits its fixed Model382 dependency receipt")
        for relative, expected in locked_files.items():
            path = REPO_ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Model392 lock/source hash mismatch: {relative}")
        return receipt

    locked_files = {
        receipt["protocol"]["path"]: receipt["protocol"]["sha256"],
        **receipt["candidate_source_sha256"],
        receipt["synthetic_smoke"]["path"]: receipt["synthetic_smoke"]["sha256"],
    }
    analysis_source = receipt.get("analysis_source_sha256")
    analysis_smoke = receipt.get("analysis_smoke_sha256")
    execution_source = receipt.get("execution_source_sha256")
    dependencies = receipt.get("analysis_dependency_sha256")
    lock_generator = receipt.get("lock_generator")
    if not analysis_source or not analysis_smoke or not execution_source or not dependencies:
        raise ValueError("Model392 evaluator, smoke, recorder, or dependencies are not locked")
    if not lock_generator or not lock_generator.get("path") or not lock_generator.get("sha256"):
        raise ValueError("Model392 pre-score lock generator is not locked")
    locked_files.update(analysis_source)
    locked_files.update(analysis_smoke)
    locked_files.update(execution_source)
    locked_files.update(dependencies)
    locked_files[lock_generator["path"]] = lock_generator["sha256"]

    runner = receipt["reference_oof_runner"]
    runner_name = Path(runner["path"]).name
    if runner_name != "run_model382_seed_count_control_p100.sh":
        raise ValueError("unexpected locked Model382 reference runner")
    locked_files[runner_name] = runner["sha256"]
    for relative, expected in locked_files.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model392 lock/source hash mismatch: {relative}")
    return receipt


def modal_width_from_training_split(images_dir: Path) -> int:
    widths = []
    for path in sorted(images_dir.glob("*_image.png")):
        with Image.open(path) as image:
            widths.append(image.width)
    if not widths:
        raise FileNotFoundError(f"no training images found in {images_dir}")
    counts = Counter(widths)
    maximum = max(counts.values())
    return min(width for width, count in counts.items() if count == maximum)


def image_id_widths(images_dir: Path) -> dict[str, int]:
    result = {}
    for path in sorted(images_dir.glob("*_image.png")):
        if not GROUP_RE.match(path.name.removesuffix("_image.png")):
            raise ValueError(f"cannot parse specimen group from {path.name}")
        with Image.open(path) as image:
            result[path.name.removesuffix("_image.png")] = int(image.width)
    if not result:
        raise FileNotFoundError(f"no images found in {images_dir}")
    return result


def verify_fold_geometry(cv_root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    metadata = receipt.get("geometry_audit", {})
    relative = metadata.get("path")
    expected_hash = metadata.get("sha256")
    if not relative or not expected_hash:
        raise ValueError("Model392 amended lock has no locked fold geometry audit")
    path = REPO_ROOT / relative
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError("Model392 fold geometry result is missing or differs from its lock")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scope") != "fixed train grouped split image filenames and native widths only":
        raise ValueError("unexpected Model392 geometry audit scope")

    for fold in range(1, 5):
        train = image_id_widths(cv_root / f"train_fold{fold}" / "images")
        heldout = image_id_widths(cv_root / f"oof_fold{fold}" / "images")
        train_groups = {GROUP_RE.match(name).group(1) for name in train}
        heldout_groups = {GROUP_RE.match(name).group(1) for name in heldout}
        if len(train_groups) == 0 or len(heldout_groups) == 0:
            raise ValueError(f"cannot parse specimen groups in fold {fold}")
        if train_groups & heldout_groups:
            raise ValueError(f"train/OOF specimen overlap in fold {fold}")
        canonical = "\n".join(
            f"{split}/{stem}_image.png\t{width}"
            for split, mapping in (("train", train), ("heldout", heldout))
            for stem, width in sorted(mapping.items())
        ).encode()
        record = result.get("folds", {}).get(str(fold), {})
        target = modal_width_from_training_split(cv_root / f"train_fold{fold}" / "images")
        if record.get("image_id_width_manifest_sha256") != hashlib.sha256(canonical).hexdigest():
            raise ValueError(f"Model392 fold {fold} train/OOF image manifest changed after audit")
        if record.get("target_width_from_training_mode") != target:
            raise ValueError(f"Model392 fold {fold} train-only target width differs from audit")
        if record.get("group_overlap") != []:
            raise ValueError(f"Model392 geometry audit recorded group overlap in fold {fold}")
    return result


def expected_training_args(root: Path, run_dir: Path, seed: int, fold: int) -> dict[str, Any]:
    root = root.resolve()
    run_dir = run_dir.resolve()
    data_root = root / f"data_model382_seed{seed}_fold{fold}"
    target_width = modal_width_from_training_split(data_root / "train" / "images")
    return {
        "data_root": str(data_root),
        "output_dir": str(run_dir),
        "device": "cuda:0",
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet",
        "patch_size": 512,
        "input_scale": 2.0,
        "normalize_width": True,
        "target_width": target_width,
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
        "rng_mode": "legacy",
        "val_every": 0,
        "no_amp": False,
        "resume": True,
    }


def lock_timestamp(lock_path: Path) -> float:
    value = json.loads(lock_path.read_text(encoding="utf-8"))["locked_at_utc"]
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Model392 lock timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def verify_candidate_manifest(
    root: Path,
    run_dir: Path,
    oof_dir: Path,
    seed: int,
    fold: int,
    model392_lock: Path,
    lock_sha256: str,
) -> Path:
    archive = oof_dir / "valid_probabilities.npz"
    manifest_path = oof_dir / "model392_manifest.json"
    checkpoint = run_dir / "last.pt"
    args_path = run_dir / "args.json"
    run_receipt_path = run_dir / "model392_run_receipt.json"
    if not all(path.is_file() for path in (archive, manifest_path, checkpoint, args_path, run_receipt_path)):
        raise FileNotFoundError(f"missing Model392 checkpoint/args/OOF manifest: {oof_dir}")
    locked_at = lock_timestamp(model392_lock)
    if any(path.stat().st_mtime <= locked_at
           for path in (archive, manifest_path, checkpoint, args_path, run_receipt_path)):
        raise ValueError(f"Model392 artifact predates its pre-score lock: {oof_dir}")

    expected_args = expected_training_args(root, run_dir, seed, fold)
    actual_args = json.loads(args_path.read_text(encoding="utf-8"))
    if actual_args != expected_args:
        different = {
            key: (actual_args.get(key), value)
            for key, value in expected_args.items()
            if actual_args.get(key) != value
        }
        raise ValueError(f"Model392 training args mismatch in {args_path}: {different}")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    args_hash = hashlib.sha256(
        json.dumps(expected_args, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (run_receipt.get("model") != 392 or run_receipt.get("seed") != seed
            or run_receipt.get("fold") != fold
            or run_receipt.get("model392_lock_sha256") != lock_sha256
            or run_receipt.get("training_args_sha256") != args_hash
            or not run_receipt.get("attempts")):
        raise ValueError(f"Model392 run receipt mismatch: {run_receipt_path}")
    initialized = datetime.fromisoformat(
        run_receipt["initialized_at_utc"].replace("Z", "+00:00")
    )
    if initialized.tzinfo is None or initialized.astimezone(timezone.utc).timestamp() <= locked_at:
        raise ValueError(f"Model392 run receipt has a naive/pre-lock timestamp: {run_receipt_path}")
    previous_epoch = 0
    for attempt_index, attempt in enumerate(run_receipt["attempts"]):
        started = datetime.fromisoformat(attempt["started_at_utc"].replace("Z", "+00:00"))
        resume_epoch = int(attempt["resume_from_epoch"])
        if (started.tzinfo is None or started.astimezone(timezone.utc).timestamp() <= locked_at
                or resume_epoch < previous_epoch or resume_epoch >= 70
                or (attempt_index == 0 and resume_epoch != 0)
                or (resume_epoch > 0 and not attempt.get("previous_checkpoint_sha256"))):
            raise ValueError(f"Model392 run receipt resume history is invalid: {run_receipt_path}")
        previous_epoch = resume_epoch

    inference = {
        **BASE_INFERENCE,
        "target_width": expected_args["target_width"],
        "scale_rule": "base_input_scale * target_width / native image width",
        "normalize_width": True,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "model": 392,
        "seed": seed,
        "fold": fold,
        "epochs": 70,
        "rng_mode": "legacy",
        "metric": METRIC,
        "target_width": expected_args["target_width"],
        "checkpoint_sha256": sha256(checkpoint),
        "oof_probabilities_sha256": sha256(archive),
        "model392_lock_sha256": lock_sha256,
        "run_receipt_sha256": sha256(run_receipt_path),
        "resume_attempts": run_receipt["attempts"],
        "inference": inference,
        "training_args": expected_args,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Model392 manifest mismatch {manifest_path}: {key}")
    return archive


def load_paired_rows(
    cv_root: Path,
    model382_root: Path,
    model392_root: Path,
    model382_lock: Path,
    model392_lock: Path,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, dict[str, Any]]]:
    cv_root = cv_root.resolve()
    model382_root = model382_root.resolve()
    model392_root = model392_root.resolve()
    receipt = verify_model392_lock(model392_lock)
    fixed_model382_lock = REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json"
    if model382_lock.resolve() != fixed_model382_lock.resolve():
        raise ValueError("Model392 is bound to the fixed Model382 preregistration receipt")
    dep_key = str(fixed_model382_lock.relative_to(REPO_ROOT))
    if sha256(model382_lock) != receipt["analysis_dependency_sha256"][dep_key]:
        raise ValueError("Model382 preregistration receipt hash differs from the Model392 lock")

    primary_receipt = json.loads(model382_lock.read_text(encoding="utf-8"))
    locked_at = float(primary_receipt["locked_at_unix"])
    primary_hashes = primary_receipt.get("artifacts_sha256", {})
    for relative in LOCKED_ARTIFACTS:
        path = REPO_ROOT / relative
        if not path.is_file() or primary_hashes.get(relative) != sha256(path):
            raise ValueError(f"Model382 pre-result source lock mismatch: {relative}")

    model392_lock_sha = sha256(model392_lock)
    rows = {seed: {} for seed in seeds}
    seen: set[str] = set()
    group_fold: dict[str, int] = {}
    project_root = model392_root.parents[1]
    verify_fold_geometry(cv_root, receipt)
    target_width_by_fold = {
        fold: expected_training_args(
            project_root, model392_root / "seed1" / f"fold{fold}", 1, fold
        )["target_width"]
        for fold in range(1, 5)
    }

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
            reference_manifest = json.loads((reference_dir / "model382_manifest.json").read_text())
            candidate_run = model392_root / f"seed{seed}/fold{fold}"
            candidate_oof = model392_root / f"seed{seed}/fold{fold}_oof"
            expected_train_path = str((cv_root / f"train_fold{fold}").resolve())
            expected_heldout_path = str((cv_root / f"oof_fold{fold}").resolve())
            if (reference_manifest.get("train_images") != expected_train_path
                    or reference_manifest.get("heldout_images") != expected_heldout_path):
                raise ValueError(f"Model382 manifest split mismatch: seed{seed}/fold{fold}")
            data_root = project_root / f"data_model382_seed{seed}_fold{fold}"
            if ((data_root / "train").resolve() != (cv_root / f"train_fold{fold}").resolve()
                    or (data_root / "valid").resolve() != (cv_root / f"oof_fold{fold}").resolve()):
                raise ValueError(f"Model392 train/held-out split mismatch: seed{seed}/fold{fold}")
            candidate_archive = verify_candidate_manifest(
                project_root,
                candidate_run,
                candidate_oof,
                seed,
                fold,
                model392_lock,
                model392_lock_sha,
            )

            reference = _load_archive(reference_archive, expected_ids)
            candidate = _load_archive(candidate_archive, expected_ids)
            try:
                for image_id in sorted(expected_ids):
                    if seed == seeds[0] and image_id in seen:
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
                    with Image.open(masks[image_id]) as mask_image:
                        native_width = int(mask_image.width)
                    target_width = target_width_by_fold[fold]
                    reference_iou = class_ious(
                        np.asarray(reference[image_id], dtype=np.float32), truth
                    )
                    candidate_iou = class_ious(
                        np.asarray(candidate[image_id], dtype=np.float32), truth
                    )
                    if seed == seeds[0]:
                        seen.add(image_id)
                    rows[seed][image_id] = {
                        "group": group,
                        "fold": fold,
                        "step": int(step_match.group(1)),
                        "native_width": native_width,
                        "target_width": target_width,
                        "reference_class_iou": reference_iou,
                        "candidate_class_iou": candidate_iou,
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


def paired_group_bootstrap(
    rows: dict[int, dict[str, dict[str, Any]]],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Resample specimen groups once, keeping both seed deltas paired per image."""
    seeds = tuple(sorted(rows))
    if seeds != (1, 2):
        raise ValueError("paired final bootstrap requires seeds 1 and 2")
    names = sorted(rows[1])
    if names != sorted(rows[2]) or len(names) != 70:
        raise ValueError("paired final bootstrap requires the same 70 images for both seeds")

    image_delta = np.asarray([
        np.mean([
            np.mean(rows[run_seed][name]["candidate_class_iou"])
            - np.mean(rows[run_seed][name]["reference_class_iou"])
            for run_seed in seeds
        ])
        for name in names
    ], dtype=np.float64)
    group_by_fold: dict[int, dict[str, list[int]]] = {}
    for fold in range(1, 5):
        group_by_fold[fold] = {}
    for index, name in enumerate(names):
        metadata = rows[1][name]
        if any(rows[2][name].get(key) != metadata.get(key)
               for key in ("group", "fold", "step", "native_width", "target_width")):
            raise ValueError(f"seed metadata mismatch for {name}")
        group_by_fold[int(metadata["fold"])].setdefault(metadata["group"], []).append(index)

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_images: list[int] = []
        for fold in range(1, 5):
            groups = sorted(group_by_fold[fold])
            if not groups:
                raise ValueError(f"fold {fold} has no specimen groups")
            sampled = rng.integers(0, len(groups), size=len(groups))
            for group_index in sampled:
                sampled_images.extend(group_by_fold[fold][groups[int(group_index)]])
        bootstrap[draw] = image_delta[sampled_images].mean()
    return np.percentile(bootstrap * 100.0, [2.5, 97.5]).astype(float).tolist()


def summarize_final_paired(
    rows: dict[int, dict[str, dict[str, Any]]],
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Summarize seeds equally and bootstrap groups while preserving seed pairing."""
    if set(rows) != {1, 2}:
        raise ValueError("final stage requires paired OOF for seeds 1 and 2")
    per_seed = {run_seed: summarize_seed(rows[run_seed], draws) for run_seed in (1, 2)}
    names = sorted(rows[1])
    if names != sorted(rows[2]) or len(names) != 70:
        raise ValueError("final analysis requires identical coverage of 70 OOF images")
    for name in names:
        for field in ("group", "fold", "step", "native_width", "target_width"):
            if rows[1][name][field] != rows[2][name][field]:
                raise ValueError(f"seed metadata mismatch for {name}: {field}")

    ref = np.asarray([
        [rows[run_seed][name]["reference_class_iou"] for name in names]
        for run_seed in (1, 2)
    ], dtype=np.float64)
    candidate = np.asarray([
        [rows[run_seed][name]["candidate_class_iou"] for name in names]
        for run_seed in (1, 2)
    ], dtype=np.float64)
    seed_image_delta = candidate.mean(axis=2) - ref.mean(axis=2)
    pooled_delta = seed_image_delta.mean(axis=0)
    ci = paired_group_bootstrap(rows, draws)
    class_delta = (candidate - ref).mean(axis=(0, 1)) * 100.0
    folds = np.asarray([int(rows[1][name]["fold"]) for name in names])
    fold_delta = {
        str(fold): float(pooled_delta[folds == fold].mean() * 100.0)
        for fold in range(1, 5)
    }
    step4 = np.asarray([int(rows[1][name]["step"]) == 4 for name in names])
    if not step4.any():
        raise ValueError("Step4 subset is empty")
    step4_delta = float(pooled_delta[step4].mean() * 100.0)
    pooled_positive_folds = sum(value > 0.0 for value in fold_delta.values())
    per_seed_safety = all(
        item["delta_pp"] > 0.0
        and sum(value > 0 for value in item["fold_delta_pp"].values()) >= 3
        and min(item["fold_delta_pp"].values()) >= -0.30
        and all(value >= -0.50 for value in item["class_delta_pp"].values())
        and item["step4_delta_pp"] >= -0.50
        for item in per_seed.values()
    )
    pooled_safety = (
        pooled_positive_folds >= 3
        and min(fold_delta.values()) >= -0.30
        and bool(np.all(class_delta >= -0.50))
        and step4_delta >= -0.50
    )
    gate = {
        "both_seed_point_deltas_positive": all(per_seed[s]["delta_pp"] > 0 for s in (1, 2)),
        "equal_weight_mean_gain_at_least_0_50pp": float(pooled_delta.mean() * 100.0) >= 0.50,
        "paired_group_bootstrap_ci_lower_above_zero": ci[0] > 0.0,
        "each_seed_fold_class_step4_safety_passed": per_seed_safety,
        "pooled_three_of_four_folds_improve": pooled_positive_folds >= 3,
        "pooled_fold_class_step4_safety_passed": pooled_safety,
    }
    return {
        "per_seed": per_seed,
        "equal_weight_delta_pp": float(pooled_delta.mean() * 100.0),
        "equal_weight_group_bootstrap_95ci_delta_pp": ci,
        "equal_weight_class_delta_pp": {
            name: float(value) for name, value in zip(CLASS_NAMES, class_delta)
        },
        "equal_weight_fold_delta_pp": fold_delta,
        "seed2_minus_seed1_delta_pp": float(
            per_seed[2]["delta_pp"] - per_seed[1]["delta_pp"]
        ),
        "equal_weight_step4_delta_pp": step4_delta,
        "bootstrap_pairing": "average seed1/seed2 image deltas first; resample each fold's specimen groups once",
        "width_strata_descriptive_only": summarize_width_strata(rows, names, seed_image_delta),
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def summarize_width_strata(
    rows: dict[int, dict[str, dict[str, Any]]],
    names: list[str],
    seed_image_delta: np.ndarray,
) -> dict[str, Any]:
    widths = np.asarray([int(rows[1][name]["native_width"]) for name in names])
    targets = np.asarray([int(rows[1][name]["target_width"]) for name in names])
    result: dict[str, Any] = {}
    for width in sorted(set(widths.tolist())):
        selected = widths == width
        selected_groups = {rows[1][name]["group"] for index, name in enumerate(names) if selected[index]}
        result[str(width)] = {
            "images": int(selected.sum()),
            "groups": len(selected_groups),
            "seed_delta_pp": {
                str(seed): float(seed_image_delta[seed_index, selected].mean() * 100.0)
                for seed_index, seed in enumerate((1, 2))
            },
            "equal_weight_delta_pp": float(seed_image_delta[:, selected].mean() * 100.0),
        }
    target_control = widths == targets
    result["native_width_equals_fold_train_mode"] = {
        "images": int(target_control.sum()),
        "groups": len({rows[1][name]["group"] for index, name in enumerate(names)
                        if target_control[index]}),
        "folds": sorted({int(rows[1][name]["fold"]) for index, name in enumerate(names)
                          if target_control[index]}),
        "seed_delta_pp": {
            str(seed): float(seed_image_delta[seed_index, target_control].mean() * 100.0)
            if target_control.any() else None
            for seed_index, seed in enumerate((1, 2))
        },
        "equal_weight_delta_pp": float(seed_image_delta[:, target_control].mean() * 100.0)
        if target_control.any() else None,
        "interpretation": "secondary descriptive subset only; not a causal or gate result",
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt = verify_model392_lock(args.model392_lock)
    lock_sha = sha256(args.model392_lock)
    output_path = args.output.resolve()
    claim_path, claim_sha = claim_confirmatory_analysis(output_path, lock_sha)
    rows = load_paired_rows(
        args.cv_root,
        args.model382_root,
        args.model392_root,
        args.model382_lock,
        args.model392_lock,
        (1, 2),
    )
    summary = summarize_final_paired(rows, BOOTSTRAP_DRAWS)
    geometry = verify_fold_geometry(args.cv_root, receipt)
    summary["fold_target_width_counts"] = {
        fold: {
            "target_width": item["target_width_from_training_mode"],
            "training_width_counts": item["training_width_counts"],
            "heldout_width_counts": item["heldout_width_counts"],
            "heldout_images_at_target_width": item["heldout_target_width_control_images"],
        }
        for fold, item in geometry["folds"].items()
    }
    result = {
        "model": 392,
        "stage": "two_seed_confirmatory",
        "scope": "70 train-only grouped OOF images / 59 specimen groups; both seeds complete before analysis",
        "metric": METRIC,
        "model392_lock_sha256": lock_sha,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "analysis_attempt_claim_sha256": claim_sha,
        "summary": summary,
        "valid_evaluation_authorized": summary["gate_passed"],
    }
    result["sources"] = {
        "model392_lock": str(args.model392_lock.resolve()),
        "model382_lock": str(args.model382_lock.resolve()),
        "cv_root": str(args.cv_root.resolve()),
        "model382_root": str(args.model382_root.resolve()),
        "model392_root": str(args.model392_root.resolve()),
        "model392_status": receipt["status"],
        "analysis_claim": str(claim_path.resolve()),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(json.dumps({"stage": result["stage"], "summary": summary,
                      "valid_evaluation_authorized": result["valid_evaluation_authorized"]},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model382-root", type=Path, required=True)
    parser.add_argument("--model392-root", type=Path, required=True)
    parser.add_argument("--model382-lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model382_e5_preregistration_lock.json")
    parser.add_argument("--model392-lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model392_widthnorm_runtime_lock.json")
    parser.add_argument("--output", type=Path, default=CONFIRMATORY_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
