#!/usr/bin/env python3
"""Immutable Model409 source, calibration, and train-OOF artifact locks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import model396_fixed_architecture_oof as m396
import model406_component_hardnegative_v3_lock as m406
import model407_component_fp_lock as m407


SOURCE_LOCK = "experiments/results/model409_instance_metric_source_lock_v2.json"
CALIBRATION = "outputs/model409_instance_metric_v2/calibration.json"
CALIBRATION_LOCK = "experiments/results/model409_instance_metric_calibration_lock_v2.json"
OOF_LOCK = "experiments/results/model409_instance_metric_oof_lock_v2.json"
OUTPUT_ROOT = "outputs/model409_instance_metric_v2/candidate"
SPLIT_ROOT = "outputs/model403_local_supcon/splits/model392_cv"
MODEL392_ROOT = "outputs/model392_widthnorm/seed1"
MODEL406_CONTROL_ROOT = "outputs/model406_component_hardnegative_v3/control"
METRIC_SCOPE = "70 train images/59 specimen groups; official valid/test excluded"

SOURCE_FILES = (
    "experiments/model409_instance_metric_preregistration_v2.md",
    "experiments/model409_instance_metric_loss.py",
    "experiments/model409_instance_metric_loss_v2.py",
    "experiments/model409_instance_metric_train.py",
    "experiments/model409_instance_metric_calibrate.py",
    "experiments/model409_instance_metric_lock.py",
    "experiments/score_model409_instance_metric_oof.py",
    "experiments/run_model409_instance_metric_oof.sh",
    "experiments/model406_component_hardnegative_v3_lock.py",
    "experiments/model407_component_fp_lock.py",
    "experiments/score_model406_component_hardnegative_oof.py",
    "experiments/score_model404_component_recall_oof_v6.py",
    "experiments/score_model404_component_recall_oof_v5.py",
    "experiments/score_model404_component_recall_oof_v4.py",
    "model397_reproduction_bundle/model392/evaluate_image_mean.py",
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
    records = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return records


def reference_model406(root: Path) -> dict:
    reference = m407.verify_model406_reference(root)
    if not (root / m406.OOF_LOCK).is_file():
        raise FileNotFoundError(root / m406.OOF_LOCK)
    return reference


def create_source(root: Path) -> dict:
    reference = reference_model406(root)
    payload = {
        "experiment": "Model409 tiny-component local feature metric source",
        "revision": 2,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric_scope": METRIC_SCOPE,
        "reference_model406": reference,
        "locked_files": file_inventory(root, SOURCE_FILES),
    }
    write_new(root / SOURCE_LOCK, payload)
    return payload


def verify_source(root: Path) -> dict:
    reference = reference_model406(root)
    record = json.loads((root / SOURCE_LOCK).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model409 tiny-component local feature metric source"
            or record.get("revision") != 2
            or record.get("metric_scope") != METRIC_SCOPE
            or record.get("reference_model406") != reference
            or record.get("locked_files") != file_inventory(root, SOURCE_FILES)):
        raise ValueError("Model409 source-lock mismatch")
    return record


def validate_calibration(root: Path) -> dict:
    record = json.loads((root / CALIBRATION).read_text(encoding="utf-8"))
    folds = record.get("folds", {})
    if (record.get("experiment") != "Model409 v2 tiny-component local feature metric calibration"
            or record.get("valid_read") is not False
            or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0
            or record.get("probe_examples_per_fold") != 32
            or set(folds) != {"1", "2", "3", "4"}):
        raise ValueError("Model409 calibration identity/scope mismatch")
    for fold, item in folds.items():
        if (item.get("training_split_only") is not True
                or item.get("optimizer_steps") != 0
                or item.get("valid_read") is not False
                or item.get("test_read") is not False
                or item.get("probe_examples") != 32
                or item.get("active_pair_crops", 0) < 24
                or len(item.get("active_aux_to_main_decoder_gradient_norm_ratios", [])) < 24
                or not 1e-5 <= float(item.get("metric_weight", 0.0)) <= 1.0):
            raise ValueError(f"unsafe Model409 fold calibration: {fold}")
    return record


def create_calibration_lock(root: Path) -> dict:
    source = verify_source(root)
    calibration = validate_calibration(root)
    payload = {
        "experiment": "Model409 v2 outer-train-only metric calibration lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "reference_model406_oof_lock_sha256": source["reference_model406"]["oof_lock_sha256"],
        "calibration_sha256": sha256(root / CALIBRATION),
        "fold_metric_weights": {
            fold: float(item["metric_weight"]) for fold, item in calibration["folds"].items()
        },
        "valid_read": False,
        "test_read": False,
        "optimizer_steps": 0,
    }
    write_new(root / CALIBRATION_LOCK, payload)
    return payload


def verify_calibration(root: Path) -> dict:
    source = verify_source(root)
    calibration = validate_calibration(root)
    record = json.loads((root / CALIBRATION_LOCK).read_text(encoding="utf-8"))
    weights = {fold: float(item["metric_weight"]) for fold, item in calibration["folds"].items()}
    if (record.get("experiment") != "Model409 v2 outer-train-only metric calibration lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("reference_model406_oof_lock_sha256") != source["reference_model406"]["oof_lock_sha256"]
            or record.get("calibration_sha256") != sha256(root / CALIBRATION)
            or record.get("fold_metric_weights") != weights
            or record.get("valid_read") is not False
            or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0):
        raise ValueError("Model409 calibration-lock mismatch")
    return record


def candidate_artifacts(root: Path) -> dict:
    calibration = validate_calibration(root)
    masks_by_fold = m396.mask_files(root)
    output_root = root / OUTPUT_ROOT
    artifacts = {}
    for fold, mask_map in masks_by_fold.items():
        split = root / SPLIT_ROOT / f"seed1_fold{fold}"
        parent = root / MODEL392_ROOT / f"fold{fold}/last.pt"
        run = output_root / f"fold{fold}"
        oof = output_root / f"fold{fold}_oof"
        paths = (
            run / "args.json",
            run / "last.pt",
            oof / "valid_probabilities.npz",
            oof / "valid_summary.json",
            oof / "valid_per_image.csv",
        )
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(next(path for path in paths if not path.is_file()))
        args = json.loads(paths[0].read_text(encoding="utf-8"))
        expected_weight = float(calibration["folds"][str(fold)]["metric_weight"])
        if (args.get("arm") != "candidate" or args.get("epochs") != 8 or args.get("seed") != 1
                or Path(args.get("init_checkpoint", "")).resolve() != parent.resolve()
                or Path(args.get("data_root", "")).resolve() != split.resolve()
                or not np.isclose(float(args.get("metric_weight", -1)), expected_weight, rtol=0, atol=1e-12)):
            raise ValueError(f"Model409 candidate recipe mismatch in fold {fold}")
        with np.load(paths[2], allow_pickle=False) as archive:
            if set(archive.files) != set(mask_map):
                raise ValueError(f"Model409 OOF IDs mismatch in fold {fold}")
        for path in paths:
            name = path.resolve().relative_to(root.resolve()).as_posix()
            artifacts[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return artifacts


def create_oof_lock(root: Path) -> dict:
    source = verify_source(root)
    calibration = verify_calibration(root)
    reference = reference_model406(root)
    m406.verify_oof(root)
    payload = {
        "experiment": "Model409 v2 candidate grouped train-OOF lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "calibration_lock_sha256": sha256(root / CALIBRATION_LOCK),
        "reference_model406_oof_lock_sha256": reference["oof_lock_sha256"],
        "reference_model406_control_oof_sha256": {
            f"fold{fold}": sha256(root / MODEL406_CONTROL_ROOT / f"fold{fold}_oof/valid_probabilities.npz")
            for fold in range(1, 5)
        },
        "candidate_locked_files": candidate_artifacts(root),
        "metric_scope": METRIC_SCOPE,
        "official_valid_and_test_used": False,
    }
    if calibration.get("valid_read") or calibration.get("test_read"):
        raise ValueError("Model409 calibration unexpectedly accessed valid/test")
    write_new(root / OOF_LOCK, payload)
    return payload


def verify_oof_lock(root: Path) -> dict:
    verify_source(root)
    calibration = verify_calibration(root)
    reference = reference_model406(root)
    m406.verify_oof(root)
    record = json.loads((root / OOF_LOCK).read_text(encoding="utf-8"))
    control_hashes = {
        f"fold{fold}": sha256(root / MODEL406_CONTROL_ROOT / f"fold{fold}_oof/valid_probabilities.npz")
        for fold in range(1, 5)
    }
    if (record.get("experiment") != "Model409 v2 candidate grouped train-OOF lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("calibration_lock_sha256") != sha256(root / CALIBRATION_LOCK)
            or record.get("reference_model406_oof_lock_sha256") != reference["oof_lock_sha256"]
            or record.get("reference_model406_control_oof_sha256") != control_hashes
            or record.get("candidate_locked_files") != candidate_artifacts(root)
            or record.get("metric_scope") != METRIC_SCOPE
            or record.get("official_valid_and_test_used") is not False
            or calibration.get("valid_read") is not False
            or calibration.get("test_read") is not False):
        raise ValueError("Model409 OOF lock mismatch")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--create-source-lock", action="store_true")
    modes.add_argument("--verify-source-lock", action="store_true")
    modes.add_argument("--create-calibration-lock", action="store_true")
    modes.add_argument("--verify-calibration-lock", action="store_true")
    modes.add_argument("--create-oof-lock", action="store_true")
    modes.add_argument("--verify-oof-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.create_source_lock:
        record, path = create_source(root), root / SOURCE_LOCK
    elif args.verify_source_lock:
        record, path = verify_source(root), root / SOURCE_LOCK
    elif args.create_calibration_lock:
        record, path = create_calibration_lock(root), root / CALIBRATION_LOCK
    elif args.verify_calibration_lock:
        record, path = verify_calibration(root), root / CALIBRATION_LOCK
    elif args.create_oof_lock:
        record, path = create_oof_lock(root), root / OOF_LOCK
    else:
        record, path = verify_oof_lock(root), root / OOF_LOCK
    print(json.dumps({"lock": str(path), "sha256": sha256(path), "locked_files": len(record)}, indent=2))


if __name__ == "__main__":
    main()
