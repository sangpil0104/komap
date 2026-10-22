#!/usr/bin/env python3
"""Create/append a lock-bound receipt immediately before a Model392 train attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from model392_widthnorm_oof import (
    REPO_ROOT,
    expected_training_args,
    sha256,
    verify_model392_lock,
)


def canonical_json_sha256(value: dict) -> str:
    import hashlib
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prepare(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    cv_root = root / "data_model322_group_cv"
    lock = args.lock.resolve()
    lock_record = verify_model392_lock(lock)
    lock_hash = sha256(lock)
    lock_dt = datetime.fromisoformat(lock_record["locked_at_utc"].replace("Z", "+00:00"))
    if lock_dt.tzinfo is None:
        raise ValueError("Model392 lock timestamp must include a timezone")
    lock_time = lock_dt.astimezone(timezone.utc).timestamp()
    model382_runner_lock = root / "outputs/model382_seed_count_control/.model382.lock"
    if model382_runner_lock.exists():
        raise RuntimeError(
            "Model382's locked P100 run is still active; refusing to prepare a concurrent GPU run"
        )

    fold = args.fold
    seed = args.seed
    data_root = root / f"data_model382_seed{seed}_fold{fold}"
    if (data_root / "train").resolve() != (cv_root / f"train_fold{fold}").resolve():
        raise ValueError("training symlink is not the locked grouped train fold")
    if (data_root / "valid").resolve() != (cv_root / f"oof_fold{fold}").resolve():
        raise ValueError("held-out symlink is not the locked grouped OOF fold")

    run_dir = root / "outputs/model392_widthnorm" / f"seed{seed}" / f"fold{fold}"
    if run_dir.is_symlink():
        raise ValueError(f"refusing a symlinked Model392 output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "model392_run_receipt.json"
    state_path = run_dir / "train_state.pt"
    checkpoint_path = run_dir / "last.pt"
    args_path = run_dir / "args.json"
    train_args = expected_training_args(root, run_dir, seed, fold)
    args_hash = canonical_json_sha256(train_args)
    now = utc_now()

    if not receipt_path.exists():
        existing = list(run_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"unreceipted Model392 output directory is not empty; refusing to adopt: {run_dir}"
            )
        if state_path.exists() or checkpoint_path.exists() or args_path.exists():
            raise FileExistsError("unreceipted checkpoint/args found; refusing to resume")
        receipt = {
            "model": 392,
            "seed": seed,
            "fold": fold,
            "model392_lock_sha256": lock_hash,
            "initialized_at_utc": now,
            "training_args_sha256": args_hash,
            "attempts": [],
        }
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("model") != 392 or receipt.get("seed") != seed
                or receipt.get("fold") != fold
                or receipt.get("model392_lock_sha256") != lock_hash
                or receipt.get("training_args_sha256") != args_hash):
            raise ValueError("existing Model392 run receipt does not match this lock/fold/config")
        if datetime.fromisoformat(receipt["initialized_at_utc"].replace("Z", "+00:00")).timestamp() <= lock_time:
            raise ValueError("Model392 run receipt predates the amended pre-score lock")
        if args_path.exists() and json.loads(args_path.read_text(encoding="utf-8")) != train_args:
            raise ValueError("existing args.json differs from the locked training configuration")
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            epoch = int(state.get("epoch", -1))
            if epoch < 0 or epoch >= 70 or not checkpoint_path.is_file():
                raise ValueError(f"invalid resumable state at epoch {epoch}")
            resume_from = epoch
            previous_checkpoint_sha = sha256(checkpoint_path)
        else:
            if checkpoint_path.exists():
                raise ValueError("last.pt exists without a resumable train_state.pt")
            resume_from = 0
            previous_checkpoint_sha = None

    if not receipt["attempts"]:
        resume_from = 0
        previous_checkpoint_sha = None
    receipt["attempts"].append({
        "started_at_utc": now,
        "resume_from_epoch": resume_from,
        "previous_checkpoint_sha256": previous_checkpoint_sha,
    })
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print(json.dumps({
        "seed": seed,
        "fold": fold,
        "resume_from_epoch": resume_from,
        "lock_sha256": lock_hash,
        "run_receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path,
                        default=REPO_ROOT / "experiments/results/model392_widthnorm_runtime_lock.json")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
