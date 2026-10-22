#!/usr/bin/env python3
"""Validate and record one completed Model392 grouped-OOF fold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model382_oof_ensemble_compare import _load_archive, class_ious, mask_to_ids
from model392_widthnorm_oof import (
    BASE_INFERENCE,
    METRIC,
    REPO_ROOT,
    expected_training_args,
    sha256,
    verify_model392_lock,
)


def record_fold(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    cv_root = root / "data_model322_group_cv"
    run_dir = root / "outputs/model392_widthnorm" / f"seed{args.seed}" / f"fold{args.fold}"
    oof_dir = root / "outputs/model392_widthnorm" / f"seed{args.seed}" / f"fold{args.fold}_oof"
    lock = args.lock.resolve()
    receipt = verify_model392_lock(lock)
    lock_hash = sha256(lock)
    lock_dt = datetime.fromisoformat(receipt["locked_at_utc"].replace("Z", "+00:00"))
    if lock_dt.tzinfo is None:
        raise ValueError("Model392 lock timestamp must include a timezone")
    lock_time = lock_dt.astimezone(timezone.utc).timestamp()

    data_root = root / f"data_model382_seed{args.seed}_fold{args.fold}"
    expected_train = cv_root / f"train_fold{args.fold}"
    expected_valid = cv_root / f"oof_fold{args.fold}"
    if (data_root / "train").resolve() != expected_train.resolve():
        raise ValueError(f"unexpected train-fold link: {data_root / 'train'}")
    if (data_root / "valid").resolve() != expected_valid.resolve():
        raise ValueError(f"unexpected held-out train-fold link: {data_root / 'valid'}")

    args_path = run_dir / "args.json"
    state_path = run_dir / "train_state.pt"
    checkpoint = run_dir / "last.pt"
    run_receipt_path = run_dir / "model392_run_receipt.json"
    train_log_path = run_dir / "train.log"
    archive_path = oof_dir / "valid_probabilities.npz"
    summary_path = oof_dir / "valid_summary.json"
    manifest_path = oof_dir / "model392_manifest.json"
    required = (args_path, state_path, checkpoint, run_receipt_path,
                train_log_path, archive_path, summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Model392 fold artifacts: {missing}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing fold manifest: {manifest_path}")
    if any(path.stat().st_mtime <= lock_time for path in required):
        raise ValueError("one or more Model392 fold artifacts predate its pre-score lock")

    expected_args = expected_training_args(root, run_dir, args.seed, args.fold)
    actual_args = json.loads(args_path.read_text(encoding="utf-8"))
    if actual_args != expected_args:
        raise ValueError(f"Model392 training arguments mismatch: {args_path}")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(run_receipt["initialized_at_utc"].replace("Z", "+00:00"))
    if started.tzinfo is None or started.astimezone(timezone.utc).timestamp() <= lock_time:
        raise ValueError("Model392 run receipt has a naive or pre-lock start timestamp")
    if (run_receipt.get("model") != 392 or run_receipt.get("seed") != args.seed
            or run_receipt.get("fold") != args.fold
            or run_receipt.get("model392_lock_sha256") != lock_hash):
        raise ValueError("Model392 run receipt does not match this lock/seed/fold")
    canonical_args = json.dumps(expected_args, sort_keys=True, separators=(",", ":")).encode()
    expected_args_hash = hashlib.sha256(canonical_args).hexdigest()
    attempts = run_receipt.get("attempts")
    if run_receipt.get("training_args_sha256") != expected_args_hash or not attempts:
        raise ValueError("Model392 run receipt args hash or attempt history is invalid")
    prior_epoch = 0
    for index, item in enumerate(attempts):
        attempt_dt = datetime.fromisoformat(item["started_at_utc"].replace("Z", "+00:00"))
        if attempt_dt.tzinfo is None or attempt_dt.astimezone(timezone.utc).timestamp() <= lock_time:
            raise ValueError("Model392 attempt history has a naive or pre-lock timestamp")
        resume_epoch = int(item["resume_from_epoch"])
        if (resume_epoch < prior_epoch or resume_epoch >= 70
                or (index == 0 and resume_epoch != 0)
                or (resume_epoch > 0 and not item.get("previous_checkpoint_sha256"))):
            raise ValueError("Model392 resume history is inconsistent")
        prior_epoch = resume_epoch
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != 70:
        raise ValueError(f"training is not complete: expected epoch 70, got {state.get('epoch')}")
    train_log = train_log_path.read_text(encoding="utf-8", errors="replace")
    if "done. final model:" not in train_log:
        raise ValueError(f"training log does not show clean completion: {train_log_path}")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_preprocess = {
        "patch_size": 512,
        "input_scale": 2.0,
        "target_width": expected_args["target_width"],
        "normalize_width": True,
    }
    if checkpoint_data.get("preprocessing_config") != expected_preprocess:
        raise ValueError("checkpoint preprocessing metadata differs from its training split")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_inference = summary.get("inference", {})
    if summary.get("split") != "valid" or summary.get("images") is None:
        raise ValueError("unexpected held-out OOF inference summary")
    if summary.get("image_miou_empty_zero") is None:
        raise ValueError("held-out summary does not report the official empty=0 metric")
    expected_summary_inference = {
        "patch_size": 512,
        "base_input_scale": 2.0,
        "target_width": expected_args["target_width"],
        "overlap": 170,
        "overlap_rule": "preserve overlap/native-tile ratio",
        "tta": "4-way horizontal/vertical flips",
        "tta_mode": "flip4",
        "tta_enabled": True,
        "downsample_antialias": False,
        "batch_size": 2,
        "device": "cuda:0",
    }
    if any(summary_inference.get(key) != value for key, value in expected_summary_inference.items()):
        raise ValueError("OOF inference settings differ from the preregistered configuration")

    mask_dir = expected_valid / "masks"
    masks = {path.name.removesuffix("_mask.png"): path for path in mask_dir.glob("*_mask.png")}
    archive = _load_archive(archive_path, set(masks))
    try:
        for image_id, mask_path in masks.items():
            truth = mask_to_ids(mask_path)
            class_ious(np.asarray(archive[image_id], dtype=np.float32), truth)
    finally:
        archive.close()

    inference = {
        **BASE_INFERENCE,
        "target_width": expected_args["target_width"],
        "scale_rule": "base_input_scale * target_width / native image width",
        "normalize_width": True,
    }
    record = {
        "model": 392,
        "seed": args.seed,
        "fold": args.fold,
        "epochs": 70,
        "rng_mode": "legacy",
        "metric": METRIC,
        "target_width": expected_args["target_width"],
        "checkpoint_sha256": sha256(checkpoint),
        "oof_probabilities_sha256": sha256(archive_path),
        "model392_lock_sha256": lock_hash,
        "run_receipt_sha256": sha256(run_receipt_path),
        "resume_attempts": run_receipt["attempts"],
        "inference": inference,
        "training_args": expected_args,
    }
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seed": args.seed,
        "fold": args.fold,
        "target_width": expected_args["target_width"],
        "oof_images": len(masks),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "oof_probabilities_sha256": record["oof_probabilities_sha256"],
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model392_widthnorm_runtime_lock.json")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    record_fold(parser.parse_args())


if __name__ == "__main__":
    main()
