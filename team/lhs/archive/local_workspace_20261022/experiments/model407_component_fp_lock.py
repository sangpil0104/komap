#!/usr/bin/env python3
"""Source, calibration, and grouped-OOF locks for Model407."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import model396_fixed_architecture_oof as m396
import model406_component_hardnegative_v2_lock as m406v2
import model406_component_hardnegative_v3_lock as m406v3


SOURCE_LOCK = "experiments/results/model407_component_fp_source_lock.json"
CALIBRATION = "outputs/model407_component_fp/calibration.json"
CALIBRATION_LOCK = "experiments/results/model407_component_fp_calibration_lock.json"
OOF_LOCK = "experiments/results/model407_component_fp_oof_lock.json"
RESULT = "experiments/results/model407_component_fp_oof.json"
CLAIM = RESULT + ".claim"
OUTPUT_ROOT = "outputs/model407_component_fp"
SPLIT_ROOT = "outputs/model403_local_supcon/splits/model392_cv"
MODEL392_ROOT = "outputs/model392_widthnorm/seed1"
M406_CONTROL_ROOT = "outputs/model406_component_hardnegative_v3/control"
M406_CANDIDATE_ROOT = "outputs/model406_component_hardnegative_v3/candidate"
M406_RESULT = "experiments/results/model406_component_hardnegative_oof_v3.json"
M406_CLAIM = M406_RESULT + ".claim"

SOURCE_FILES = (
    "experiments/model407_component_fp_preregistration.md",
    "experiments/model407_component_fp_loss.py",
    "experiments/model407_component_fp_train.py",
    "experiments/model407_component_fp_calibrate.py",
    "experiments/model407_component_fp_lock.py",
    "experiments/model407_tiny_component_audit.py",
    "experiments/run_model407_component_fp_oof.sh",
    "experiments/score_model407_component_fp_oof.py",
    "experiments/model404_component_recall_train.py",
    "experiments/model404_component_recall_loss_v4.py",
    "experiments/model406_component_hardnegative_v3_lock.py",
    "experiments/score_model406_component_hardnegative_oof.py",
    "experiments/model406_tiny_component_confusion_audit.py",
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


def verify_model406_reference(root: Path) -> dict:
    m406v3.verify_oof(root)
    result_path = root / M406_RESULT
    claim_path = root / M406_CLAIM
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    expected = {
        "source_lock_sha256": sha256(root / m406v3.SOURCE_LOCK),
        "calibration_lock_sha256": sha256(root / m406v2.CALIBRATION_LOCK),
        "oof_lock_sha256": sha256(root / m406v3.OOF_LOCK),
        "claim_sha256": sha256(claim_path),
    }
    if (result.get("experiment") != "Model406 v3 small-component recall with hard-negative island suppression"
            or result.get("status") != "NO_GO"
            or result.get("valid_read") is not False
            or result.get("test_read") is not False
            or any(result.get(key) != value for key, value in expected.items())
            or claim.get("oof_lock_sha256") != expected["oof_lock_sha256"]):
        raise ValueError("Model406 reference result/claim/lock chain mismatch")
    return {
        **expected,
        "result_sha256": sha256(result_path),
        "claim_sha256": sha256(claim_path),
    }


def create_source(root: Path) -> dict:
    reference = verify_model406_reference(root)
    payload = {
        "experiment": "Model407 prediction-component FP suppression source",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "reference_model406": reference,
        "reference_oof_lock_sha256": reference["oof_lock_sha256"],
        "locked_files": file_inventory(root, SOURCE_FILES),
    }
    write_new(root / SOURCE_LOCK, payload)
    return payload


def verify_source(root: Path) -> dict:
    reference = verify_model406_reference(root)
    record = json.loads((root / SOURCE_LOCK).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model407 prediction-component FP suppression source"
            or record.get("revision") != 1
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"
            or record.get("reference_model406") != reference
            or record.get("locked_files") != file_inventory(root, SOURCE_FILES)):
        raise ValueError("Model407 source lock mismatch")
    return record


def validate_calibration(root: Path) -> dict:
    record = json.loads((root / CALIBRATION).read_text(encoding="utf-8"))
    folds = record.get("folds", {})
    if (record.get("experiment") != "Model407 GT recall plus prediction-component FP calibration"
            or record.get("valid_read") is not False
            or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0
            or record.get("probe_examples_per_fold") != 32
            or set(folds) != {"1", "2", "3", "4"}):
        raise ValueError("Model407 calibration identity/scope mismatch")
    for fold, item in folds.items():
        if (item.get("training_split_only") is not True
                or item.get("optimizer_steps") != 0
                or item.get("valid_read") is not False
                or item.get("test_read") is not False
                or item.get("probe_examples") != 32
                or item.get("nonempty_aux_gradient_examples", 0) < 24
                or not 1e-5 <= float(item.get("joint_aux_weight", 0.0)) <= 1.0
                or len(item.get("joint_aux_to_main_decoder_gradient_norm_ratios", [])) < 24):
            raise ValueError(f"unsafe Model407 fold calibration: {fold}")
    return record


def create_calibration_lock(root: Path) -> dict:
    source = verify_source(root)
    calibration = validate_calibration(root)
    payload = {
        "experiment": "Model407 outer-train-only component-FP calibration lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "reference_model406_oof_lock_sha256": source["reference_oof_lock_sha256"],
        "calibration_sha256": sha256(root / CALIBRATION),
        "calibration_scope": "32 crops/fold, outer-train only, optimizer steps=0",
        "fold_aux_weights": {
            fold: float(item["joint_aux_weight"])
            for fold, item in calibration["folds"].items()
        },
        "valid_read": False,
        "test_read": False,
    }
    write_new(root / CALIBRATION_LOCK, payload)
    return payload


def verify_calibration(root: Path) -> dict:
    source = verify_source(root)
    calibration = validate_calibration(root)
    record = json.loads((root / CALIBRATION_LOCK).read_text(encoding="utf-8"))
    expected_weights = {
        fold: float(item["joint_aux_weight"])
        for fold, item in calibration["folds"].items()
    }
    if (record.get("experiment") != "Model407 outer-train-only component-FP calibration lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("reference_model406_oof_lock_sha256") != source["reference_oof_lock_sha256"]
            or record.get("calibration_sha256") != sha256(root / CALIBRATION)
            or record.get("fold_aux_weights") != expected_weights
            or record.get("valid_read") is not False
            or record.get("test_read") is not False):
        raise ValueError("Model407 calibration lock mismatch")
    return record


def candidate_artifacts(root: Path) -> dict[str, dict]:
    calibration = validate_calibration(root)
    masks_by_fold = m396.mask_files(root)
    output_root = root / OUTPUT_ROOT / "candidate"
    artifacts: dict[str, dict] = {}
    for fold, mask_map in masks_by_fold.items():
        split = root / SPLIT_ROOT / f"seed1_fold{fold}"
        parent = root / MODEL392_ROOT / f"fold{fold}" / "last.pt"
        run = output_root / f"fold{fold}"
        oof = output_root / f"fold{fold}_oof"
        args_path = run / "args.json"
        checkpoint = run / "last.pt"
        probability = oof / "valid_probabilities.npz"
        summary = oof / "valid_summary.json"
        per_image = oof / "valid_per_image.csv"
        for path in (args_path, checkpoint, probability, summary, per_image):
            if not path.is_file():
                raise FileNotFoundError(path)
        args = json.loads(args_path.read_text(encoding="utf-8"))
        expected_weight = float(calibration["folds"][str(fold)]["joint_aux_weight"])
        if (args.get("arm") != "candidate"
                or args.get("epochs") != 8
                or args.get("seed") != 1
                or Path(args.get("init_checkpoint", "")).resolve() != parent.resolve()
                or Path(args.get("data_root", "")).resolve() != split.resolve()
                or not np.isclose(float(args.get("component_weight", -1)), expected_weight,
                                  rtol=0.0, atol=1e-12)):
            raise ValueError(f"Model407 candidate recipe mismatch in fold {fold}")
        with np.load(probability, allow_pickle=False) as archive:
            if set(archive.files) != set(mask_map):
                raise ValueError(f"Model407 OOF IDs mismatch in fold {fold}")
        for path in (args_path, checkpoint, probability, summary, per_image):
            name = path.resolve().relative_to(root.resolve()).as_posix()
            artifacts[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return artifacts


def create_oof_lock(root: Path) -> dict:
    verify_calibration(root)
    reference = verify_model406_reference(root)
    candidate = candidate_artifacts(root)
    payload = {
        "experiment": "Model407 candidate grouped train-OOF lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "calibration_lock_sha256": sha256(root / CALIBRATION_LOCK),
        "reference_model406_oof_lock_sha256": reference["oof_lock_sha256"],
        "reference_model406_result_sha256": reference["result_sha256"],
        "reference_control_oof_sha256": {
            f"fold{fold}": m406v3.sha256(
                root / M406_CONTROL_ROOT / f"fold{fold}_oof" / "valid_probabilities.npz"
            )
            for fold in range(1, 5)
        },
        "candidate_locked_files": candidate,
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "official_valid_and_test_used": False,
    }
    write_new(root / OOF_LOCK, payload)
    return payload


def verify_oof_lock(root: Path) -> dict:
    verify_calibration(root)
    reference = verify_model406_reference(root)
    record = json.loads((root / OOF_LOCK).read_text(encoding="utf-8"))
    expected_control = {
        f"fold{fold}": m406v3.sha256(
            root / M406_CONTROL_ROOT / f"fold{fold}_oof" / "valid_probabilities.npz"
        )
        for fold in range(1, 5)
    }
    if (record.get("experiment") != "Model407 candidate grouped train-OOF lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("calibration_lock_sha256") != sha256(root / CALIBRATION_LOCK)
            or record.get("reference_model406_oof_lock_sha256") != reference["oof_lock_sha256"]
            or record.get("reference_model406_result_sha256") != reference["result_sha256"]
            or record.get("reference_control_oof_sha256") != expected_control
            or record.get("candidate_locked_files") != candidate_artifacts(root)
            or record.get("official_valid_and_test_used") is not False
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"):
        raise ValueError("Model407 OOF lock mismatch")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create-source-lock", action="store_true")
    group.add_argument("--verify-source-lock", action="store_true")
    group.add_argument("--create-calibration-lock", action="store_true")
    group.add_argument("--verify-calibration-lock", action="store_true")
    group.add_argument("--create-oof-lock", action="store_true")
    group.add_argument("--verify-oof-lock", action="store_true")
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
    print(json.dumps({
        "lock": str(path),
        "sha256": sha256(path),
        "locked_files": len(record.get("locked_files", record.get("candidate_locked_files", {}))),
    }, indent=2))


if __name__ == "__main__":
    main()
