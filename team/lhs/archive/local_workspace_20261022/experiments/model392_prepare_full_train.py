#!/usr/bin/env python3
"""Prepare one lock-bound Model392 full-train seed after the two-seed OOF gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from model392_official_valid_once import (
    full_train_args,
    image_mask_stems,
    verify_oof_gate,
)
from model392_widthnorm_oof import REPO_ROOT, sha256, verify_model392_lock


def canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def prepare(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    lock_path = args.model392_lock.resolve()
    verify_model392_lock(lock_path)
    lock_hash = sha256(lock_path)
    gate_path = root / "experiments/results/model392_widthnorm_two_seed_confirmatory.json"
    verify_oof_gate(gate_path, lock_hash)
    gate_hash = sha256(gate_path)

    model382_lock = root / "outputs/model382_seed_count_control/.model382.lock"
    if model382_lock.exists():
        raise RuntimeError("Model382 P100 run is still locked; refusing concurrent full-train work")

    train_root = root / "data/train"
    train_ids, _ = image_mask_stems(train_root)
    if len(train_ids) != 70:
        raise ValueError(f"expected 70 official train images, found {len(train_ids)}")
    data_root = root / "data_model392_full"
    if data_root.exists() and not data_root.is_dir():
        raise ValueError(f"full-train data-root path is not a directory: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    train_link = data_root / "train"
    if not train_link.exists() and not train_link.is_symlink():
        train_link.symlink_to(train_root, target_is_directory=True)
    if not train_link.is_dir() or train_link.resolve() != train_root.resolve():
        raise ValueError("data_model392_full/train must resolve exactly to data/train")

    seed = args.seed
    run_dir = root / "outputs/model392_widthnorm/full_train" / f"seed{seed}"
    if run_dir.is_symlink():
        raise ValueError(f"refusing symlinked full-train output directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "model392_full_train_receipt.json"
    state_path = run_dir / "train_state.pt"
    checkpoint_path = run_dir / "last.pt"
    args_path = run_dir / "args.json"
    from model392_widthnorm_oof import modal_width_from_training_split
    target_width = modal_width_from_training_split(train_root / "images")
    expected = full_train_args(root, seed, target_width)
    args_hash = canonical_sha256(expected)
    now = utc_now()
    already_complete = False
    needs_finalize = False

    if not receipt_path.exists():
        existing = [path for path in run_dir.iterdir() if path.name != receipt_path.name]
        if existing:
            raise FileExistsError(f"unreceipted full-train directory is not empty: {run_dir}")
        receipt = {
            "model": 392,
            "stage": "full_train",
            "seed": seed,
            "model392_lock_sha256": lock_hash,
            "oof_gate_sha256": gate_hash,
            "initialized_at_utc": now,
            "training_args_sha256": args_hash,
            "attempts": [],
        }
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("model") != 392 or receipt.get("stage") != "full_train"
                or receipt.get("seed") != seed
                or receipt.get("model392_lock_sha256") != lock_hash
                or receipt.get("oof_gate_sha256") != gate_hash
                or receipt.get("training_args_sha256") != args_hash):
            raise ValueError("existing full-train receipt does not match the authorized recipe")
        if args_path.exists() and json.loads(args_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("existing full-train args.json differs from the locked recipe")
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            epoch = int(state.get("epoch", -1))
            if epoch < 0 or epoch > 70 or state.get("args") != expected:
                raise ValueError(f"invalid full-train resume state at epoch {epoch}")
            if not checkpoint_path.is_file():
                raise FileNotFoundError("full-train resume state has no inference checkpoint")
            resume_from = epoch
            previous_checkpoint_hash = sha256(checkpoint_path)
            if epoch == 70:
                if not receipt["attempts"]:
                    raise ValueError("completed full-train state has no recorded training attempt")
                log_path = run_dir / "train.log"
                clean_end = log_path.is_file() and "done. final model:" in log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                already_complete = bool(clean_end)
                needs_finalize = not already_complete
        else:
            if checkpoint_path.exists():
                raise ValueError("full-train last.pt exists without resumable train_state.pt")
            resume_from = 0
            previous_checkpoint_hash = None

    if not receipt["attempts"]:
        resume_from = 0
        previous_checkpoint_hash = None
    if not needs_finalize and not already_complete:
        receipt["attempts"].append({
            "started_at_utc": now,
            "resume_from_epoch": resume_from,
            "previous_checkpoint_sha256": previous_checkpoint_hash,
        })
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
    print(json.dumps({
        "seed": seed,
        "resume_from_epoch": resume_from,
        "already_complete": already_complete,
        "needs_finalize": needs_finalize,
        "run_dir": str(run_dir),
        "model392_lock_sha256": lock_hash,
        "oof_gate_sha256": gate_hash,
        "training_args": expected,
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--model392-lock", type=Path,
        default=REPO_ROOT / "experiments/results/model392_widthnorm_runtime_lock.json",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
