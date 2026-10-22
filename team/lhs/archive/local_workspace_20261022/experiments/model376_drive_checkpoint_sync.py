#!/usr/bin/env python3
"""Safely mirror resumable grouped-OOF checkpoints to mounted Google Drive.

Training remains on Colab's local disk for throughput. Snapshots are immutable
epoch directories and contain a matching inference checkpoint and full resume
state. Only the newest two snapshots are retained per fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


SNAPSHOT_RE = re.compile(r"snapshot_epoch_(\d{4})$")
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s*/\s*\d+\s+train_loss=")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_epoch_and_check(checkpoint: Path, state_path: Path) -> int:
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = int(state["epoch"])
    resume_weights = state["model"]
    infer_weights = checkpoint_obj["state_dict"]
    if resume_weights.keys() != infer_weights.keys():
        raise ValueError("checkpoint and resumable state have different model keys")
    for key in resume_weights:
        left, right = resume_weights[key], infer_weights[key]
        if hasattr(left, "shape") and hasattr(right, "shape"):
            if left.shape != right.shape or not torch.equal(left, right):
                raise ValueError(f"checkpoint pair weights differ at {key}")
        elif left != right:
            raise ValueError(f"checkpoint pair values differ at {key}")
    del state, checkpoint_obj
    return epoch


def manifest_for(folder: Path, model_id: int | None = None) -> dict[str, Any]:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if model_id is not None and int(manifest.get("model", -1)) != model_id:
        raise ValueError(f"snapshot model id differs: expected {model_id}, found {manifest.get('model')}")
    checkpoint = folder / "last.pt"
    state_path = folder / "train_state.pt"
    if checkpoint.stat().st_size != manifest["last_pt_bytes"]:
        raise ValueError("last.pt size differs from snapshot manifest")
    if state_path.stat().st_size != manifest["train_state_bytes"]:
        raise ValueError("train_state.pt size differs from snapshot manifest")
    if sha256_file(checkpoint) != manifest["last_pt_sha256"]:
        raise ValueError("last.pt checksum mismatch")
    if sha256_file(state_path) != manifest["train_state_sha256"]:
        raise ValueError("train_state.pt checksum mismatch")
    epoch = pair_epoch_and_check(checkpoint, state_path)
    if epoch != int(manifest["epoch"]):
        raise ValueError("snapshot epoch differs from manifest")
    return manifest


def snapshot(run_dir: Path, drive_dir: Path, expected_epoch: int | None = None,
             keep: int = 2, model_id: int = 376) -> Path:
    checkpoint = run_dir / "last.pt"
    state_path = run_dir / "train_state.pt"
    for path in (checkpoint, state_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"checkpoint file missing or empty: {path}")

    before = (checkpoint.stat(), state_path.stat())
    epoch = pair_epoch_and_check(checkpoint, state_path)
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(f"expected epoch {expected_epoch}, found {epoch}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    final_dir = drive_dir / f"snapshot_epoch_{epoch:04d}"

    if final_dir.exists():
        existing = manifest_for(final_dir, model_id)
        current_checkpoint_hash = sha256_file(checkpoint)
        current_state_hash = sha256_file(state_path)
        if (existing["last_pt_sha256"] != current_checkpoint_hash or
                existing["train_state_sha256"] != current_state_hash):
            raise FileExistsError(f"refusing to overwrite a different snapshot: {final_dir}")
        log(f"snapshot already present and verified: {final_dir}")
        return final_dir

    stage = drive_dir / f".staging_epoch_{epoch:04d}_{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        shutil.copy2(checkpoint, stage / "last.pt")
        shutil.copy2(state_path, stage / "train_state.pt")
        log_path = run_dir / "train.log"
        if log_path.is_file():
            shutil.copy2(log_path, stage / "train.log")

        after = (checkpoint.stat(), state_path.stat())
        if any((a.st_size, a.st_mtime_ns) != (b.st_size, b.st_mtime_ns)
               for a, b in zip(before, after)):
            raise RuntimeError("source checkpoint changed while Drive snapshot was copied")

        copied_epoch = pair_epoch_and_check(stage / "last.pt", stage / "train_state.pt")
        if copied_epoch != epoch:
            raise RuntimeError("copied checkpoint pair changed epoch")
        manifest = {
            "model": model_id,
            "fold": int(re.search(r"fold(\d+)$", run_dir.name).group(1))
                    if re.search(r"fold(\d+)$", run_dir.name) else None,
            "epoch": epoch,
            "last_pt_bytes": (stage / "last.pt").stat().st_size,
            "train_state_bytes": (stage / "train_state.pt").stat().st_size,
            "last_pt_sha256": sha256_file(stage / "last.pt"),
            "train_state_sha256": sha256_file(stage / "train_state.pt"),
            "created_unix": time.time(),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(stage, final_dir)
        log(f"Drive checkpoint snapshot verified: model={model_id} fold={manifest['fold']} epoch={epoch}")

        snapshots = []
        for child in drive_dir.iterdir():
            match = SNAPSHOT_RE.fullmatch(child.name)
            if match and child.is_dir():
                snapshots.append((int(match.group(1)), child))
        snapshots.sort(reverse=True)
        for _, old_dir in snapshots[keep:]:
            # Prune only our own immutable snapshot directories in this fold's
            # dedicated output folder; never traverse or touch other Drive data.
            shutil.rmtree(old_dir)
            log(f"pruned older Model{model_id} snapshot: {old_dir.name}")
        return final_dir
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def restore(run_dir: Path, drive_dir: Path, model_id: int = 376) -> int:
    candidates = []
    if drive_dir.is_dir():
        for child in drive_dir.iterdir():
            match = SNAPSHOT_RE.fullmatch(child.name)
            if match and child.is_dir():
                candidates.append((int(match.group(1)), child))
    candidates.sort(reverse=True)
    if not candidates:
        log(f"no Drive checkpoint found for {run_dir.name}; starting fresh")
        return 0

    local_epoch = 0
    try:
        local_epoch = pair_epoch_and_check(run_dir / "last.pt", run_dir / "train_state.pt")
    except (OSError, KeyError, ValueError, RuntimeError):
        pass

    for folder_epoch, folder in candidates:
        try:
            manifest = manifest_for(folder, model_id)
            if int(manifest["epoch"]) != folder_epoch:
                raise ValueError("snapshot directory name does not match its manifest")
            if local_epoch >= folder_epoch:
                log(f"keeping newer verified local checkpoint at epoch {local_epoch}")
                return local_epoch

            stage = Path(tempfile.mkdtemp(prefix=f".model{model_id}-restore-", dir=run_dir.parent))
            try:
                shutil.copy2(folder / "last.pt", stage / "last.pt")
                shutil.copy2(folder / "train_state.pt", stage / "train_state.pt")
                pair_epoch = pair_epoch_and_check(stage / "last.pt", stage / "train_state.pt")
                if pair_epoch != folder_epoch:
                    raise ValueError("restored checkpoint pair epoch mismatch")
                run_dir.mkdir(parents=True, exist_ok=True)
                os.replace(stage / "last.pt", run_dir / "last.pt")
                os.replace(stage / "train_state.pt", run_dir / "train_state.pt")
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            log(f"restored verified Model{model_id} Drive checkpoint pair: {folder.name}")
            return folder_epoch
        except Exception as exc:
            log(f"ignoring invalid Drive snapshot {folder.name}: {type(exc).__name__}: {exc}")
    if local_epoch:
        log(f"no valid Drive snapshot; using local epoch {local_epoch}")
        return local_epoch
    raise RuntimeError(f"no valid resumable Model{model_id} snapshot in {drive_dir}")


def latest_logged_epoch(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    text = log_path.read_text(errors="replace")
    epochs = [int(match.group(1)) for match in EPOCH_RE.finditer(text)]
    return epochs[-1] if epochs else 0


def watch(run_dir: Path, drive_dir: Path, interval: float, stable_seconds: float,
          keep: int, model_id: int = 376) -> None:
    log_path = run_dir / "train.log"
    previous_signature = None
    last_attempt_signature = None
    stable_since = 0.0
    last_snapshot_epoch = 0
    while True:
        try:
            completed_epoch = latest_logged_epoch(log_path)
            checkpoint = run_dir / "last.pt"
            state_path = run_dir / "train_state.pt"
            if checkpoint.is_file() and state_path.is_file():
                stats = (checkpoint.stat(), state_path.stat())
                signature = tuple((item.st_size, item.st_mtime_ns) for item in stats)
                if signature != previous_signature:
                    previous_signature = signature
                    stable_since = time.monotonic()
                due = completed_epoch > 0 and (completed_epoch % 5 == 0 or completed_epoch == 70)
                if due and completed_epoch > last_snapshot_epoch and (
                        signature != last_attempt_signature and
                        time.monotonic() - stable_since >= stable_seconds):
                    last_attempt_signature = signature
                    try:
                        snapshot(run_dir, drive_dir, expected_epoch=completed_epoch,
                                 keep=keep, model_id=model_id)
                        last_snapshot_epoch = completed_epoch
                    except Exception as exc:
                        log(f"snapshot attempt for epoch {completed_epoch} deferred: "
                            f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            log(f"watch poll failed; will retry: {type(exc).__name__}: {exc}")
        time.sleep(interval)


def finalize(run_dir: Path, oof_dir: Path, drive_fold_dir: Path,
             model_id: int = 376) -> None:
    snapshot(run_dir, drive_fold_dir / "checkpoints", expected_epoch=70,
             keep=2, model_id=model_id)
    if not oof_dir.is_dir():
        raise FileNotFoundError(f"OOF output directory missing: {oof_dir}")
    final_oof = drive_fold_dir / "oof"
    if final_oof.exists():
        log(f"Drive OOF folder already exists; preserving it: {final_oof}")
    else:
        stage = drive_fold_dir / f".staging_oof_{uuid.uuid4().hex}"
        shutil.copytree(oof_dir, stage)
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        marker = {
            "model": model_id,
            "fold": int(re.search(r"fold(\d+)", oof_dir.name).group(1)),
            "kind": "grouped_train_oof_not_official_validation",
            "files": {path.relative_to(stage).as_posix(): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)
            } for path in files},
        }
        (stage / "drive_manifest.json").write_text(json.dumps(marker, indent=2) + "\n")
        os.rename(stage, final_oof)
        log(f"saved and checksummed grouped OOF artifacts to Drive: {final_oof}")

    for source, name in (
            (run_dir / "args.json", "args.json"),
            (run_dir / "train.log", "train.log"),
            (run_dir.parent / "run.log", "run.log"),
            (run_dir.parent / f"{run_dir.name}_oof_eval.log", "oof_eval.log"),
            (run_dir.parent / f"{run_dir.name}_drive_sync.log", "drive_sync.log")):
        if source.is_file():
            shutil.copy2(source, drive_fold_dir / name)
    summary = oof_dir / "valid_summary.json"
    completion = {
        "model": model_id,
        "fold": int(re.search(r"fold(\d+)", oof_dir.name).group(1)),
        "checkpoint_epoch": 70,
        "checkpoint_snapshot": str((drive_fold_dir / "checkpoints" / "snapshot_epoch_0070").name),
        "oof_summary": json.loads(summary.read_text()) if summary.is_file() else None,
        "oof_type": "grouped_train_oof_not_official_validation",
        "completed_unix": time.time(),
    }
    (drive_fold_dir / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    log(f"fold completion manifest saved: {drive_fold_dir / 'complete.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("snapshot", "restore"):
        sub = commands.add_parser(command)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--drive-dir", type=Path, required=True)
        sub.add_argument("--model-id", type=int, default=376)
        if command == "snapshot":
            sub.add_argument("--expected-epoch", type=int)
            sub.add_argument("--keep", type=int, default=2)

    sub = commands.add_parser("watch")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--drive-dir", type=Path, required=True)
    sub.add_argument("--interval", type=float, default=5.0)
    sub.add_argument("--stable-seconds", type=float, default=8.0)
    sub.add_argument("--keep", type=int, default=2)
    sub.add_argument("--model-id", type=int, default=376)

    sub = commands.add_parser("finalize")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--oof-dir", type=Path, required=True)
    sub.add_argument("--drive-fold-dir", type=Path, required=True)
    sub.add_argument("--model-id", type=int, default=376)

    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.run_dir, args.drive_dir, args.expected_epoch, args.keep, args.model_id)
    elif args.command == "restore":
        restore(args.run_dir, args.drive_dir, args.model_id)
    elif args.command == "watch":
        watch(args.run_dir, args.drive_dir, args.interval, args.stable_seconds,
              args.keep, args.model_id)
    elif args.command == "finalize":
        finalize(args.run_dir, args.oof_dir, args.drive_fold_dir, args.model_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"fatal: {type(exc).__name__}: {exc}")
        sys.exit(1)
