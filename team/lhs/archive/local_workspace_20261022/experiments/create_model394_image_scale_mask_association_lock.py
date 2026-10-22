#!/usr/bin/env python3
"""Freeze the post-hoc Model394 image-scale estimator before reading images."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ZIP = ROOT / "train.zip"
MODEL393_RESULT = ROOT / "experiments/results/model393_synthetic_mask_feasibility.json"
MODEL393_LOCK = ROOT / "experiments/results/model393_synthetic_mask_feasibility_lock.json"
PROTOCOL = "experiments/model394_image_scale_mask_association_protocol.md"
ANALYSIS = "experiments/model394_image_scale_mask_association.py"
SMOKE = "experiments/model394_image_scale_mask_association_smoke.py"
OUTPUT = ROOT / "experiments/results/model394_image_scale_mask_association_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Model394 lock: {OUTPUT}")
    paths = (TRAIN_ZIP, MODEL393_RESULT, MODEL393_LOCK)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_paths = {
        PROTOCOL: sha256(ROOT / PROTOCOL),
        ANALYSIS: sha256(ROOT / ANALYSIS),
        SMOKE: sha256(ROOT / SMOKE),
        "experiments/create_model394_image_scale_mask_association_lock.py": sha256(
            Path(__file__).resolve()
        ),
    }
    receipt = {
        "model": 394,
        "stage": "posthoc_train_only_apparent_scale_association_preanalysis_lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "train_zip_sha256": sha256(TRAIN_ZIP),
        "model393_result_sha256": sha256(MODEL393_RESULT),
        "model393_lock_sha256": sha256(MODEL393_LOCK),
        "sources_sha256": source_paths,
        "scope": "Model393 mask-summary findings were inspected before this post-hoc hypothesis; no train image pixel payload was read before the lock; valid/test/OOF/checkpoints/scores excluded",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    record = create()
    print(
        json.dumps(
            {
                "lock": str(OUTPUT),
                "locked_at_utc": record["locked_at_utc"],
                "train_zip_sha256": record["train_zip_sha256"],
                "model393_result_sha256": record["model393_result_sha256"],
                "lock_sha256": sha256(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
