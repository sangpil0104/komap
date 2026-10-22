#!/usr/bin/env python3
"""Source and grouped-OOF artifact locks for Model406 v3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import model396_fixed_architecture_oof as m396
import model406_component_hardnegative_v2_lock as v2


SOURCE_LOCK = "experiments/results/model406_component_hardnegative_source_lock_v3.json"
OOF_LOCK = "experiments/results/model406_component_hardnegative_oof_lock_v3.json"
OUTPUT_ROOT = "outputs/model406_component_hardnegative_v3"
SPLIT_ROOT = "outputs/model403_local_supcon/splits/model392_cv"
MODEL392_ROOT = "outputs/model392_widthnorm/seed1"
SOURCE_FILES = (
    v2.SOURCE_LOCK,
    v2.CALIBRATION_LOCK,
    v2.CALIBRATION,
    "experiments/model406_component_hardnegative_preregistration_v3.md",
    "experiments/run_model406_component_hardnegative_oof_v3.sh",
    "experiments/score_model406_component_hardnegative_oof_v3.py",
    "experiments/model406_component_hardnegative_v3_lock.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def file_inventory(root: Path, names: tuple[str, ...]) -> dict:
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def create_source(root: Path) -> dict:
    v2.verify_calibration(root)
    payload = {
        "experiment": "Model406 v3 paired grouped-OOF source",
        "revision": 3,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "parent_v2_source_lock_sha256": sha256(root / v2.SOURCE_LOCK),
        "v2_calibration_lock_sha256": sha256(root / v2.CALIBRATION_LOCK),
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "folds": 4,
        "arms": ["control", "candidate"],
        "locked_files": file_inventory(root, SOURCE_FILES),
    }
    write_new(root / SOURCE_LOCK, payload)
    return payload


def verify_source(root: Path) -> dict:
    v2.verify_calibration(root)
    record = json.loads((root / SOURCE_LOCK).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model406 v3 paired grouped-OOF source"
            or record.get("revision") != 3
            or record.get("parent_v2_source_lock_sha256") != sha256(root / v2.SOURCE_LOCK)
            or record.get("v2_calibration_lock_sha256") != sha256(root / v2.CALIBRATION_LOCK)
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"
            or record.get("folds") != 4):
        raise ValueError("Model406 v3 source-lock identity or parent mismatch")
    if file_inventory(root, SOURCE_FILES) != record.get("locked_files"):
        raise ValueError("Model406 v3 source inputs changed after lock")
    return record


def oof_artifacts(root: Path) -> list[Path]:
    calibration = v2.validate_calibration(root)
    masks_by_fold = m396.mask_files(root)
    output_root = root / OUTPUT_ROOT
    files: list[Path] = [root / v2.CALIBRATION, root / v2.CALIBRATION_LOCK]
    for fold, mask_map in masks_by_fold.items():
        parent = root / MODEL392_ROOT / f"fold{fold}/last.pt"
        data_root = root / SPLIT_ROOT / f"seed1_fold{fold}"
        for arm in ("control", "candidate"):
            run = output_root / arm / f"fold{fold}"
            oof = output_root / arm / f"fold{fold}_oof"
            args_path = run / "args.json"
            checkpoint = run / "last.pt"
            probability = oof / "valid_probabilities.npz"
            summary = oof / "valid_summary.json"
            per_image = oof / "valid_per_image.csv"
            for path in (args_path, checkpoint, probability, summary, per_image):
                if not path.is_file():
                    raise FileNotFoundError(path)
            train_args = json.loads(args_path.read_text(encoding="utf-8"))
            expected_weight = (float(calibration["folds"][str(fold)]["joint_aux_weight"])
                               if arm == "candidate" else 0.0)
            if (train_args.get("arm") != arm or train_args.get("epochs") != 8
                    or train_args.get("seed") != 1
                    or Path(train_args.get("init_checkpoint", "")).resolve() != parent.resolve()
                    or Path(train_args.get("data_root", "")).resolve() != data_root.resolve()
                    or not np.isclose(float(train_args.get("component_weight", -1)), expected_weight,
                                      rtol=0.0, atol=1e-12)):
                raise ValueError(f"Model406 v3 recipe mismatch: {arm}/fold{fold}")
            with np.load(probability, allow_pickle=False) as archive:
                if set(archive.files) != set(mask_map):
                    raise ValueError(f"Model406 v3 OOF image IDs mismatch: {arm}/fold{fold}")
            files.extend((args_path, checkpoint, probability, summary, per_image))
    return files


def create_oof(root: Path) -> dict:
    source = verify_source(root)
    artifacts = oof_artifacts(root)
    locked = {}
    for path in artifacts:
        name = path.resolve().relative_to(root.resolve()).as_posix()
        locked[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "experiment": "Model406 v3 paired component-correction grouped-OOF artifacts",
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "parent_v2_source_lock_sha256": source["parent_v2_source_lock_sha256"],
        "v2_calibration_lock_sha256": sha256(root / v2.CALIBRATION_LOCK),
        "official_valid_and_test_used": False,
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "locked_files": locked,
    }
    write_new(root / OOF_LOCK, payload)
    return payload


def verify_oof(root: Path) -> dict:
    verify_source(root)
    record = json.loads((root / OOF_LOCK).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model406 v3 paired component-correction grouped-OOF artifacts"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("v2_calibration_lock_sha256") != sha256(root / v2.CALIBRATION_LOCK)
            or record.get("official_valid_and_test_used") is not False
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"):
        raise ValueError("Model406 v3 OOF lock identity/scope mismatch")
    expected = {}
    for path in oof_artifacts(root):
        name = path.resolve().relative_to(root.resolve()).as_posix()
        expected[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected != record.get("locked_files"):
        raise ValueError("Model406 v3 OOF artifacts changed after lock")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-source-lock", action="store_true")
    mode.add_argument("--verify-source-lock", action="store_true")
    mode.add_argument("--create-oof-lock", action="store_true")
    mode.add_argument("--verify-oof-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.create_source_lock:
        record, path = create_source(root), root / SOURCE_LOCK
    elif args.verify_source_lock:
        record, path = verify_source(root), root / SOURCE_LOCK
    elif args.create_oof_lock:
        record, path = create_oof(root), root / OOF_LOCK
    else:
        record, path = verify_oof(root), root / OOF_LOCK
    print(json.dumps({"lock": str(path), "sha256": sha256(path),
                      "locked_files": len(record.get("locked_files", {}))}, indent=2))


if __name__ == "__main__":
    main()
