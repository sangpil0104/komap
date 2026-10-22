#!/usr/bin/env python3
"""Create/verify immutable source hashes for the Model404 pre-run protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


LOCK_REL = Path("experiments/results/model404_preregistration_lock_v3.json")
PREVIOUS_LOCK_REL = Path("experiments/results/model404_preregistration_lock_v2.json")
FILES = (
    "experiments/model404_component_recall_preregistration.md",
    "experiments/model404_component_recall_loss.py",
    "experiments/model404_component_recall_smoke.py",
    "experiments/model404_component_recall_train.py",
    "experiments/model404_component_recall_calibrate.py",
    "experiments/model404_component_recall_preflight.py",
    "experiments/score_model404_component_recall_oof.py",
    "experiments/run_model404_component_recall_oof.sh",
    "experiments/model404_preregistration_lock.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock_path = root / LOCK_REL
    if args.create:
        if lock_path.exists():
            raise FileExistsError(f"refusing to replace existing preregistration lock: {lock_path}")
        files = {}
        for relative in FILES:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            files[relative] = sha256(path)
        payload = {
            "experiment": "Model404 Eutectic-Si component-normalized recall fine-tuning",
            "revision": 3,
            "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "files_sha256": files,
            "supersedes": {
                "path": str(PREVIOUS_LOCK_REL),
                "sha256": sha256(root / PREVIOUS_LOCK_REL) if (root / PREVIOUS_LOCK_REL).is_file() else None,
            },
            "valid_read_before_lock": False,
            "test_read_before_lock": False,
            "training_started_before_lock": False,
        }
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"created {lock_path}")
    else:
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "Model404 Eutectic-Si component-normalized recall fine-tuning":
            raise ValueError("unexpected Model404 lock identity")
        if payload.get("revision") != 3:
            raise ValueError("unexpected Model404 lock revision")
        previous = payload.get("supersedes", {})
        previous_path = root / previous.get("path", "")
        if (not previous.get("sha256") or not previous_path.is_file()
                or sha256(previous_path) != previous["sha256"]):
            raise ValueError("superseded Model404 receipt is missing or changed")
        expected = payload.get("files_sha256", {})
        if set(expected) != set(FILES):
            raise ValueError("Model404 lock source inventory mismatch")
        for relative, digest in expected.items():
            path = root / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Model404 source hash mismatch: {relative}")
        print(f"verified {len(expected)} locked Model404 sources")


if __name__ == "__main__":
    main()
