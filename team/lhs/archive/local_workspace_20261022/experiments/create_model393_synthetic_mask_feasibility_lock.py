#!/usr/bin/env python3
"""Create the immutable pre-analysis lock for the Model393 mask-only screen."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ZIP = ROOT / "train.zip"
PROTOCOL = "experiments/model393_synthetic_mask_feasibility_protocol.md"
ANALYSIS = "experiments/model393_synthetic_mask_feasibility_audit.py"
SMOKE = "experiments/model393_synthetic_mask_feasibility_smoke.py"
OUTPUT = ROOT / "experiments/results/model393_synthetic_mask_feasibility_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Model393 feasibility lock: {OUTPUT}")
    if not TRAIN_ZIP.is_file():
        raise FileNotFoundError(f"training archive is missing: {TRAIN_ZIP}")
    sources = {
        PROTOCOL: sha256(ROOT / PROTOCOL),
        ANALYSIS: sha256(ROOT / ANALYSIS),
        SMOKE: sha256(ROOT / SMOKE),
        "experiments/create_model393_synthetic_mask_feasibility_lock.py": sha256(Path(__file__).resolve()),
    }
    receipt = {
        "model": 393,
        "stage": "pre_analysis_lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "train_zip_sha256": sha256(TRAIN_ZIP),
        "sources_sha256": sources,
        "scope": "train.zip masks only; no mask pixels read before lock; no image pixels, valid/test labels, OOF, checkpoints, or scores",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    record = create()
    print(json.dumps({
        "lock": str(OUTPUT),
        "locked_at_utc": record["locked_at_utc"],
        "train_zip_sha256": record["train_zip_sha256"],
        "lock_sha256": sha256(OUTPUT),
    }, ensure_ascii=False, indent=2))
