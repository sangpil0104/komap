#!/usr/bin/env python3
"""Create the pre-score secondary lock for Model382 two-seed averaging."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/model382_two_seed_mean_preregistration_lock.json"
PRIMARY_LOCK = ROOT / "experiments/results/model382_e5_preregistration_lock.json"
LOCKED_FILES = (
    "experiments/model382_two_seed_mean_sensitivity_preregistration.md",
    "experiments/model382_two_seed_mean_sensitivity.py",
    "experiments/model382_two_seed_mean_sensitivity_smoke.py",
    "experiments/model382_two_seed_mean_sensitivity_lock.py",
    "experiments/model382_seed_count_control_preregistration.md",
    "experiments/model382_five_model_ensemble_addendum.md",
    "experiments/model382_oof_ensemble_compare.py",
    "experiments/results/model382_e5_preregistration_lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {OUTPUT}")
    sources = {relative: sha256(ROOT / relative) for relative in LOCKED_FILES}
    primary_sha = sha256(PRIMARY_LOCK)
    now = time.time()
    receipt = {
        "model": 382,
        "analysis": "model382_two_seed_mean_sensitivity_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "locked_at_unix": now,
        "seed1_oof_scores_read_before_lock": False,
        "seed2_oof_scores_read_before_lock": False,
        "official_valid_or_test_read": False,
        "primary_lock_sha256": primary_sha,
        "source_sha256": sources,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
