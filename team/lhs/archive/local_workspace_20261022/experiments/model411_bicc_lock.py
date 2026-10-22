#!/usr/bin/env python3
"""Immutable provenance locks for Model411 source, calibration, and train OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

import model396_fixed_architecture_oof as m396
import model406_component_hardnegative_v3_lock as m406v3
import model410_component_sampler_lock as m410lock
from model411_bicc_loss import GT_BRANCH_WEIGHT, PRED_BRANCH_WEIGHT


SOURCE_LOCK = "experiments/results/model411_bicc_source_lock.json"
CALIBRATION = "outputs/model411_bicc/calibration.json"
CALIBRATION_LOCK = "experiments/results/model411_bicc_calibration_lock.json"
OOF_LOCK = "experiments/results/model411_bicc_oof_lock.json"
OUTPUT_ROOT = "outputs/model411_bicc"
SPLIT_ROOT = "outputs/model403_local_supcon/splits/model392_cv"
MODEL392_ROOT = "outputs/model392_widthnorm/seed1"
M406_CONTROL_ROOT = "outputs/model406_component_hardnegative_v3/control"
M406_RESULT = "experiments/results/model406_component_hardnegative_oof_v3.json"
M406_CLAIM = M406_RESULT + ".claim"
M410_RESULT = "experiments/results/model410_component_sampler_oof.json"
M410_CLAIM = M410_RESULT + ".claim"
GROUP_RE = re.compile(r"^(A\d+)_")

SOURCE_FILES = (
    "experiments/model411_bicc_preregistration.md",
    "experiments/model411_bicc_loss.py",
    "experiments/model411_bicc_train.py",
    "experiments/model411_bicc_calibrate.py",
    "experiments/model411_predict_oof_probabilities.py",
    "experiments/model411_bicc_lock.py",
    "experiments/run_model411_bicc_oof.sh",
    "experiments/queue_model411_after410.sh",
    "experiments/score_model411_bicc_oof.py",
    "experiments/model404_component_recall_train.py",
    "experiments/model404_component_recall_loss_v4.py",
    "experiments/model406_component_hardnegative_v2_lock.py",
    "experiments/model406_component_hardnegative_v3_lock.py",
    "experiments/score_model406_component_hardnegative_oof.py",
    "experiments/model410_component_sampler_lock.py",
    "experiments/model396_fixed_architecture_oof.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/inference.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def inventory(root: Path, paths: list[Path]) -> dict:
    result = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(root.resolve()).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def reference_artifacts(root: Path) -> dict:
    m406v3.verify_oof(root)
    result_path = root / M406_RESULT
    claim_path = root / M406_CLAIM
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    expected = {
        "source_lock_sha256": sha256(root / m406v3.SOURCE_LOCK),
        "calibration_lock_sha256": sha256(root / m406v3.v2.CALIBRATION_LOCK),
        "oof_lock_sha256": sha256(root / m406v3.OOF_LOCK),
        "claim_sha256": sha256(claim_path),
    }
    if (result.get("status") != "NO_GO"
            or result.get("experiment") != "Model406 v3 small-component recall with hard-negative island suppression"
            or result.get("valid_read") is not False
            or result.get("test_read") is not False
            or result.get("valid_stage_allowed") is not False
            or any(result.get(key) != value for key, value in expected.items())
            or claim.get("oof_lock_sha256") != sha256(root / m406v3.OOF_LOCK)):
        raise ValueError("Model406 v3 fixed control result/claim scope is not verified")
    m410_lock = m410lock.verify_oof_lock()
    m410_result_path = root / M410_RESULT
    m410_claim_path = root / M410_CLAIM
    m410_result = json.loads(m410_result_path.read_text(encoding="utf-8"))
    m410_claim = json.loads(m410_claim_path.read_text(encoding="utf-8"))
    m410_expected = {
        "source_lock_sha256": m410lock.sha256(m410lock.SOURCE_LOCK),
        "oof_lock_sha256": m410lock.sha256(m410lock.OOF_LOCK),
        "claim_sha256": m410lock.sha256(m410_claim_path),
    }
    if (m410_result.get("experiment") != "Model410 small-component-balanced crop sampling"
            or m410_result.get("status") != "NO_GO"
            or m410_result.get("preregistered_gate", {}).get("pass") is not False
            or m410_result.get("valid_read") is not False
            or m410_result.get("test_read") is not False
            or m410_result.get("official_valid_stage_allowed") is not False
            or any(m410_result.get(key) != value for key, value in m410_expected.items())
            or m410_claim.get("oof_lock_sha256") != m410_expected["oof_lock_sha256"]
            or m410_lock.get("training_images") != 70):
        raise ValueError("Model410 no-go result/claim/lock chain is not verified")
    parent_paths = [
        root / MODEL392_ROOT / f"fold{fold}/last.pt"
        for fold in range(1, 5)
    ] + [
        root / MODEL392_ROOT / f"fold{fold}_oof/valid_probabilities.npz"
        for fold in range(1, 5)
    ]
    control_paths = [
        root / M406_CONTROL_ROOT / f"fold{fold}_oof/valid_probabilities.npz"
        for fold in range(1, 5)
    ]
    references = inventory(root, parent_paths + control_paths)
    return {
        "model406_source_lock_sha256": sha256(root / m406v3.SOURCE_LOCK),
        "model406_oof_lock_sha256": sha256(root / m406v3.OOF_LOCK),
        "model406_result_sha256": sha256(result_path),
        "model406_claim_sha256": sha256(claim_path),
        "model410_no_go_result_sha256": sha256(m410_result_path),
        "model410_no_go_claim_sha256": sha256(m410_claim_path),
        "model410_source_lock_sha256": m410_expected["source_lock_sha256"],
        "model410_oof_lock_sha256": m410_expected["oof_lock_sha256"],
        "model392_parent_and_control_probabilities": references,
    }


def source_inventory(root: Path) -> dict:
    paths = [root / name for name in SOURCE_FILES]
    return inventory(root, paths)


def create_source(root: Path) -> dict:
    reference = reference_artifacts(root)
    payload = {
        "experiment": "Model411 component-balanced bidirectional small-island loss source",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "reference_artifacts": reference,
        "fixed_branch_weights": {
            "gt": GT_BRANCH_WEIGHT,
            "prediction": PRED_BRANCH_WEIGHT,
        },
        "locked_files": source_inventory(root),
    }
    write_new(root / SOURCE_LOCK, payload)
    return payload


def verify_source(root: Path) -> dict:
    reference = reference_artifacts(root)
    record = json.loads((root / SOURCE_LOCK).read_text(encoding="utf-8"))
    if (record.get("experiment") != "Model411 component-balanced bidirectional small-island loss source"
            or record.get("revision") != 1
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"
            or record.get("reference_artifacts") != reference
            or record.get("fixed_branch_weights") != {"gt": GT_BRANCH_WEIGHT, "prediction": PRED_BRANCH_WEIGHT}
            or record.get("locked_files") != source_inventory(root)):
        raise ValueError("Model411 source lock mismatch")
    return record


def validate_calibration(root: Path) -> dict:
    record = json.loads((root / CALIBRATION).read_text(encoding="utf-8"))
    folds = record.get("folds", {})
    if (record.get("experiment") != "Model411 component-balanced bidirectional small-island loss"
            or record.get("optimizer_steps") != 0
            or record.get("valid_read") is not False
            or record.get("test_read") is not False
            or record.get("fixed_branch_weights") != {"ground_truth_components": GT_BRANCH_WEIGHT,
                                                          "unmatched_predicted_components": PRED_BRANCH_WEIGHT}
            or set(folds) != {"1", "2", "3", "4"}):
        raise ValueError("Model411 calibration identity/scope mismatch")
    for fold, item in folds.items():
        if (item.get("optimizer_steps") != 0
                or item.get("valid_read") is not False
                or item.get("test_read") is not False
                or item.get("probe_examples") != 32
                or item.get("active_aux_gradient_examples", 0) < 24
                or not 1e-5 <= float(item.get("coefficient", 0.0)) <= 1.0
                or len(item.get("aux_to_main_decoder_gradient_norm_ratios", [])) < 24):
            raise ValueError(f"unsafe Model411 calibration for fold {fold}")
    return record


def create_calibration_lock(root: Path) -> dict:
    verify_source(root)
    calibration = validate_calibration(root)
    payload = {
        "experiment": "Model411 outer-train-only zero-step calibration lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "calibration_sha256": sha256(root / CALIBRATION),
        "fold_coefficients": {
            fold: float(item["coefficient"])
            for fold, item in calibration["folds"].items()
        },
        "probe_scope": "32 outer-train crops/fold; optimizer steps=0",
        "official_valid_and_test_used": False,
    }
    write_new(root / CALIBRATION_LOCK, payload)
    return payload


def verify_calibration(root: Path) -> dict:
    verify_source(root)
    calibration = validate_calibration(root)
    record = json.loads((root / CALIBRATION_LOCK).read_text(encoding="utf-8"))
    coefficients = {fold: float(item["coefficient"])
                    for fold, item in calibration["folds"].items()}
    if (record.get("experiment") != "Model411 outer-train-only zero-step calibration lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("calibration_sha256") != sha256(root / CALIBRATION)
            or record.get("fold_coefficients") != coefficients
            or record.get("official_valid_and_test_used") is not False):
        raise ValueError("Model411 calibration lock mismatch")
    return record


def candidate_artifacts(root: Path) -> dict:
    calibration = validate_calibration(root)
    masks_by_fold = m396.mask_files(root)
    output_root = root / OUTPUT_ROOT / "candidate"
    artifacts = {}
    total_ids: set[str] = set()
    groups: set[str] = set()
    group_fold: dict[str, int] = {}
    total_count = 0
    for fold, mask_map in masks_by_fold.items():
        total_ids.update(mask_map)
        total_count += len(mask_map)
        for image_id in mask_map:
            match = GROUP_RE.match(image_id)
            if match is None:
                raise ValueError(f"cannot parse held-out specimen group from {image_id}")
            group = match.group(1)
            previous = group_fold.setdefault(group, fold)
            if previous != fold:
                raise ValueError(f"specimen group crosses held-out folds: {group}")
            groups.add(group)
        split = root / SPLIT_ROOT / f"seed1_fold{fold}"
        parent = root / MODEL392_ROOT / f"fold{fold}/last.pt"
        run = output_root / f"fold{fold}"
        oof = output_root / f"fold{fold}_oof"
        args_path = run / "args.json"
        checkpoint_path = run / "last.pt"
        probability_path = oof / "valid_probabilities.npz"
        manifest_path = oof / "inference_manifest.json"
        for path in (args_path, checkpoint_path, probability_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if (args.get("arm") != "candidate"
                or args.get("epochs") != 8
                or args.get("seed") != 1
                or Path(args.get("init_checkpoint", "")).resolve() != parent.resolve()
                or Path(args.get("data_root", "")).resolve() != split.resolve()
                or not np.isclose(float(args.get("component_weight", -1)),
                                  float(calibration["folds"][str(fold)]["coefficient"]),
                                  rtol=0.0, atol=1e-12)):
            raise ValueError(f"Model411 training recipe mismatch for fold {fold}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {})
        preprocessing = checkpoint.get("preprocessing_config", {})
        if (model_config != {
                "architecture": "UnetPlusPlus",
                "encoder_name": "tu-hrnet_w48",
                "boundary_head": True,
        } or preprocessing.get("patch_size") != 512
                or preprocessing.get("input_scale") != 2.0
                or preprocessing.get("normalize_width") is not True):
            raise ValueError(f"Model411 checkpoint preprocessing mismatch for fold {fold}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("experiment") != "Model411 held-out train prediction-only inference"
                or manifest.get("images") != len(mask_map)
                or manifest.get("masks_opened") is not False
                or manifest.get("metrics_computed") is not False
                or manifest.get("official_valid_or_test_used") is not False
                or manifest.get("checkpoint_sha256") != sha256(checkpoint_path)
                or set(manifest.get("image_ids", [])) != set(mask_map)
                or manifest.get("inference", {}).get("patch_size") != 512
                or manifest.get("inference", {}).get("input_scale") != 2.0
                or manifest.get("inference", {}).get("target_width") != preprocessing.get("target_width")
                or manifest.get("inference", {}).get("overlap_base") != 170
                or manifest.get("inference", {}).get("tta") != "flip4"
                or manifest.get("inference", {}).get("batch_size") != 2):
            raise ValueError(f"Model411 prediction-only manifest mismatch for fold {fold}")
        with np.load(probability_path, allow_pickle=False) as archive:
            if set(archive.files) != set(mask_map):
                raise ValueError(f"Model411 held-out IDs mismatch for fold {fold}")
        for path in (args_path, checkpoint_path, probability_path, manifest_path):
            name = path.resolve().relative_to(root.resolve()).as_posix()
            artifacts[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if len(total_ids) != 70 or total_count != 70 or len(groups) != 59:
        raise ValueError(
            f"expected 70 exactly-once held-out images/59 groups; "
            f"got {len(total_ids)}/{total_count} images and {len(groups)} groups"
        )
    return artifacts


def create_oof_lock(root: Path) -> dict:
    verify_calibration(root)
    reference = reference_artifacts(root)
    candidate = candidate_artifacts(root)
    payload = {
        "experiment": "Model411 candidate grouped train-OOF lock",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_lock_sha256": sha256(root / SOURCE_LOCK),
        "calibration_lock_sha256": sha256(root / CALIBRATION_LOCK),
        "reference_model406_oof_lock_sha256": reference["model406_oof_lock_sha256"],
        "reference_probability_hashes": {
            path: value["sha256"]
            for path, value in reference["model392_parent_and_control_probabilities"].items()
        },
        "candidate_artifacts": candidate,
        "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
        "official_valid_and_test_used": False,
    }
    write_new(root / OOF_LOCK, payload)
    return payload


def verify_oof(root: Path) -> dict:
    verify_calibration(root)
    reference = reference_artifacts(root)
    record = json.loads((root / OOF_LOCK).read_text(encoding="utf-8"))
    expected_ref = {
        path: value["sha256"]
        for path, value in reference["model392_parent_and_control_probabilities"].items()
    }
    if (record.get("experiment") != "Model411 candidate grouped train-OOF lock"
            or record.get("source_lock_sha256") != sha256(root / SOURCE_LOCK)
            or record.get("calibration_lock_sha256") != sha256(root / CALIBRATION_LOCK)
            or record.get("reference_model406_oof_lock_sha256") != reference["model406_oof_lock_sha256"]
            or record.get("reference_probability_hashes") != expected_ref
            or record.get("candidate_artifacts") != candidate_artifacts(root)
            or record.get("official_valid_and_test_used") is not False
            or record.get("metric_scope") != "70 train images/59 specimen groups; official valid/test excluded"):
        raise ValueError("Model411 OOF lock mismatch")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-source-lock", action="store_true")
    mode.add_argument("--verify-source-lock", action="store_true")
    mode.add_argument("--create-calibration-lock", action="store_true")
    mode.add_argument("--verify-calibration-lock", action="store_true")
    mode.add_argument("--create-oof-lock", action="store_true")
    mode.add_argument("--verify-oof-lock", action="store_true")
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
        record, path = verify_oof(root), root / OOF_LOCK
    print(json.dumps({"lock": str(path), "sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
