#!/usr/bin/env python3
"""Immutable provenance locks for the low-memory Model406 v2 calibration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import model406_component_hardnegative_lock as v1


SOURCE_LOCK = "experiments/results/model406_component_hardnegative_source_lock_v2.json"
CALIBRATION_LOCK = "experiments/results/model406_component_hardnegative_calibration_lock_v2.json"
CALIBRATION = "outputs/model406_component_hardnegative_v2/calibration.json"
EXTRA_FILES = (
    "experiments/model406_component_hardnegative_calibrate_v2.py",
    "experiments/model406_component_hardnegative_preregistration_v2.md",
    "experiments/model406_component_hardnegative_v2_lock.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, record: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def verify_source(root: Path) -> dict:
    path = root / SOURCE_LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    v1.verify_source_lock(root)
    if (record.get("experiment") != "Model406 v2 low-memory calibration source"
            or record.get("parent_source_lock_sha256") != sha256(root / v1.SOURCE_LOCK)):
        raise ValueError("Model406 v2 source lock identity/parent mismatch")
    for name, metadata in record["locked_files"].items():
        item = root / name
        if (not item.is_file() or item.stat().st_size != metadata["size_bytes"]
                or sha256(item) != metadata["sha256"]):
            raise ValueError(f"Model406 v2 locked source changed: {name}")
    expected = set(EXTRA_FILES) | {v1.SOURCE_LOCK}
    if set(record["locked_files"]) != expected:
        raise ValueError("Model406 v2 source inventory mismatch")
    return record


def create_source(root: Path) -> dict:
    v1.verify_source_lock(root)
    files = (v1.SOURCE_LOCK, *EXTRA_FILES)
    locked = {}
    for name in files:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        locked[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    record = {
        "experiment": "Model406 v2 low-memory calibration source",
        "revision": 2,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "parent_source_lock_sha256": sha256(root / v1.SOURCE_LOCK),
        "probe_batch_size": 1,
        "probe_examples_per_fold": 32,
        "optimizer_steps": 0,
        "locked_files": locked,
    }
    write_new(root / SOURCE_LOCK, record)
    return record


def validate_calibration(root: Path) -> dict:
    record = json.loads((root / CALIBRATION).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model406 v2 component-recall plus hard-negative calibration"
            or record.get("valid_read") is not False or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0 or record.get("probe_batch_size") != 1
            or record.get("probe_examples_per_fold") != 32
            or set(record.get("folds", {})) != {"1", "2", "3", "4"}):
        raise ValueError("Model406 v2 calibration scope/schema mismatch")
    for fold, item in record["folds"].items():
        if (not item["training_split_only"] or item["optimizer_steps"] != 0
                or item["valid_read"] or item["test_read"]
                or item["probe_batch_size"] != 1 or item["probe_examples"] != 32
                or item["nonempty_gradient_examples"] < 24
                or not 1e-5 <= float(item["joint_aux_weight"]) <= 1.0):
            raise ValueError(f"unsafe Model406 v2 calibration in fold {fold}")
    return record


def create_calibration(root: Path) -> dict:
    source = verify_source(root)
    calibration = validate_calibration(root)
    path = root / CALIBRATION
    record = {
        "experiment": "Model406 v2 outer-train-only low-memory calibration",
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "valid_read": False,
        "test_read": False,
        "optimizer_steps": 0,
        "folds": sorted(calibration["folds"]),
        "locked_files": {
            CALIBRATION: {"size_bytes": path.stat().st_size, "sha256": sha256(path)},
        },
        "parent_source_lock_sha256": source["parent_source_lock_sha256"],
    }
    write_new(root / CALIBRATION_LOCK, record)
    return record


def verify_calibration(root: Path) -> dict:
    verify_source(root)
    path = root / CALIBRATION_LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model406 v2 outer-train-only low-memory calibration"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("valid_read") is not False or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0 or record.get("folds") != ["1", "2", "3", "4"]):
        raise ValueError("Model406 v2 calibration lock mismatch")
    for name, metadata in record["locked_files"].items():
        item = root / name
        if not item.is_file() or sha256(item) != metadata["sha256"]:
            raise ValueError(f"Model406 v2 calibration output changed: {name}")
    validate_calibration(root)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-source-lock", action="store_true")
    mode.add_argument("--verify-source-lock", action="store_true")
    mode.add_argument("--create-calibration-lock", action="store_true")
    mode.add_argument("--verify-calibration-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.create_source_lock:
        record = create_source(root)
        path = root / SOURCE_LOCK
    elif args.verify_source_lock:
        record = verify_source(root)
        path = root / SOURCE_LOCK
    elif args.create_calibration_lock:
        record = create_calibration(root)
        path = root / CALIBRATION_LOCK
    else:
        record = verify_calibration(root)
        path = root / CALIBRATION_LOCK
    print(json.dumps({"lock": str(path), "sha256": sha256(path),
                      "locked_files": len(record.get("locked_files", {}))}, indent=2))


if __name__ == "__main__":
    main()
