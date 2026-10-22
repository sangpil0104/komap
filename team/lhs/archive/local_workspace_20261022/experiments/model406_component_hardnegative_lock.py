#!/usr/bin/env python3
"""Create and verify immutable source, calibration, and OOF locks for Model406."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import model396_fixed_architecture_oof as m396
import model397_fixed_b4_model392_blend as m397
import model405_small_component_fixed_blend_oof as m405


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = "experiments/results/model406_component_hardnegative_source_lock_v1.json"
CALIBRATION_LOCK = "experiments/results/model406_component_hardnegative_calibration_lock_v1.json"
OOF_LOCK = "experiments/results/model406_component_hardnegative_oof_lock_v1.json"
CALIBRATION = "outputs/model406_component_hardnegative_v1/calibration.json"
OUTPUT_ROOT = "outputs/model406_component_hardnegative_v1"
RESULT = "experiments/results/model406_component_hardnegative_oof_v1.json"
CLAIM = RESULT + ".claim"
SPLIT_ROOT = "outputs/model403_local_supcon/splits/model392_cv"
MODEL404_LOCK = "experiments/results/model404_preregistration_lock_v6.json"
MODEL404_LOCK_SCRIPT = "experiments/model404_preregistration_lock_v6.py"
MODEL405_LOCK = "experiments/results/model405_small_component_fixed_blend_lock.json"
MODEL405_RESULT = "experiments/results/model405_small_component_fixed_blend_oof.json"
MODEL405_CLAIM = MODEL405_RESULT + ".claim"
MODEL392_ROOT = "outputs/model392_widthnorm/seed1"
GROUP_RE = re.compile(r"^(A\d+)_")

CODE_FILES = (
    "experiments/model406_component_hardnegative_preregistration_v1.md",
    "experiments/model406_component_hardnegative_loss.py",
    "experiments/model406_component_hardnegative_train.py",
    "experiments/model406_component_hardnegative_calibrate.py",
    "experiments/model406_component_hardnegative_lock.py",
    "experiments/run_model406_component_hardnegative_oof_v1.sh",
    "experiments/score_model406_component_hardnegative_oof.py",
    "experiments/model404_component_recall_loss_v4.py",
    "experiments/model404_component_recall_train.py",
    "experiments/model404_component_recall_calibrate.py",
    "experiments/model404_component_recall_train_v4.py",
    "experiments/model404_component_recall_calibrate_v4.py",
    "experiments/model404_preregistration_lock_v6.py",
    "experiments/score_model404_component_recall_oof.py",
    "experiments/score_model404_component_recall_oof_v4.py",
    "experiments/score_model404_component_recall_oof_v5.py",
    "experiments/score_model404_component_recall_oof_v6.py",
    "experiments/model396_fixed_architecture_oof.py",
    "experiments/model397_fixed_b4_model392_blend.py",
    "experiments/model405_small_component_fixed_blend_oof.py",
    "experiments/model405_small_component_fixed_blend_preregistration.md",
    "model397_reproduction_bundle/scripts/score_model397_oof.py",
    "model397_reproduction_bundle/scripts/write_oof_manifest.py",
    "model397_reproduction_bundle/model392/evaluate_image_mean.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/checkpoint.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/dataset.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/losses.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/model.py",
    "model336_hrnet_widthnorm_candidate/komap_seg/palette.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def verify_ancestor_experiments(root: Path) -> None:
    m397.verify_lock(root)
    m405.verify_lock(root)
    subprocess.run(
        [sys.executable, str(root / MODEL404_LOCK_SCRIPT),
         "--root", str(root), "--verify"],
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    model397_result = json.loads((root / m397.RESULT).read_text(encoding="utf-8"))
    model405_result = json.loads((root / MODEL405_RESULT).read_text(encoding="utf-8"))
    model405_claim_path = root / MODEL405_CLAIM
    model405_claim = json.loads(model405_claim_path.read_text(encoding="utf-8"))
    if (model397_result.get("valid_stage_allowed") is not False
            or model397_result.get("official_valid_used") is not False
            or model397_result.get("test_used") is not False):
        raise ValueError("Model397 parent was not a train-only OOF result")
    if (model405_result.get("status") != "NO_GO"
            or model405_result.get("official_valid_used") is not False
            or model405_result.get("test_used") is not False
            or model405_result.get("lock_sha256") != sha256(root / MODEL405_LOCK)
            or model405_result.get("claim_sha256") != sha256(model405_claim_path)
            or model405_claim.get("lock_sha256") != sha256(root / MODEL405_LOCK)):
        raise ValueError("Model405 motivation record is missing or out of scope")


def split_files(root: Path) -> tuple[list[Path], dict[int, set[str]]]:
    masks_by_fold = m396.mask_files(root)
    full_ids = {image_id for fold_masks in masks_by_fold.values() for image_id in fold_masks}
    files: list[Path] = []
    for fold in range(1, 5):
        base = root / SPLIT_ROOT / f"seed1_fold{fold}"
        train_images = base / "train/images"
        train_masks = base / "train/masks"
        valid_images = base / "valid/images"
        valid_masks = base / "valid/masks"
        for directory in (train_images, train_masks, valid_images, valid_masks):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        image_paths = sorted(train_images.glob("*_image.png"))
        mask_paths = sorted(train_masks.glob("*_mask.png"))
        heldout_image_paths = sorted(valid_images.glob("*_image.png"))
        heldout_mask_paths = sorted(valid_masks.glob("*_mask.png"))
        train_ids = {p.name.removesuffix("_image.png") for p in image_paths}
        train_mask_ids = {p.name.removesuffix("_mask.png") for p in mask_paths}
        valid_ids = {p.name.removesuffix("_image.png") for p in heldout_image_paths}
        valid_mask_ids = {p.name.removesuffix("_mask.png") for p in heldout_mask_paths}
        fixed_heldout = set(masks_by_fold[fold])
        if (train_ids != train_mask_ids or valid_ids != valid_mask_ids
                or valid_ids != fixed_heldout or train_ids & fixed_heldout
                or train_ids | fixed_heldout != full_ids):
            raise ValueError(f"Model392 CV train/heldout split mismatch in fold {fold}")
        train_groups = {GROUP_RE.match(image_id).group(1) for image_id in train_ids}
        heldout_groups = {GROUP_RE.match(image_id).group(1) for image_id in fixed_heldout}
        if train_groups & heldout_groups:
            raise ValueError(f"specimen group leaks across train/OOF fold {fold}")
        for path in image_paths + mask_paths + heldout_image_paths + heldout_mask_paths:
            resolved = path.resolve()
            resolved.relative_to((root / "data" / "train").resolve())
            files.append(resolved)
        print(
            f"verified fold {fold}: outer-train={len(train_ids)} "
            f"heldout-OOF={len(fixed_heldout)}",
            flush=True,
        )
    return sorted(set(files)), full_ids


def source_input_paths(root: Path) -> list[Path]:
    split_inputs, _ = split_files(root)
    paths = [root / name for name in CODE_FILES]
    paths.extend((
        root / MODEL404_LOCK,
        root / "experiments/results/model404_component_recall_oof_v6.json",
        root / "experiments/results/model404_predicted_component_fp_audit_v6.json",
        root / MODEL405_LOCK,
        root / MODEL405_RESULT,
        root / MODEL405_CLAIM,
        root / m397.LOCK,
        root / m397.RESULT,
        root / m397.CLAIM,
        root / "model397_reproduction_bundle/config/model397_group_folds.json",
        root / "experiments/results/model392_widthnorm_runtime_lock.json",
        root / "experiments/results/model392_widthnorm_two_seed_confirmatory.json",
        root / "experiments/results/model392_sequence_closure_lock.json",
    ))
    for fold, masks in m396.mask_files(root).items():
        paths.extend(masks.values())
        paths.extend((
            root / MODEL392_ROOT / f"fold{fold}/last.pt",
            root / MODEL392_ROOT / f"fold{fold}/args.json",
            root / MODEL392_ROOT / f"fold{fold}/model392_run_receipt.json",
            root / MODEL392_ROOT / f"fold{fold}_oof/model392_manifest.json",
            root / MODEL392_ROOT / f"fold{fold}_oof/valid_probabilities.npz",
        ))
    paths.extend(split_inputs)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def write_lock(root: Path, relative_path: str, experiment: str,
               files: list[Path], extras: dict) -> dict:
    destination = root / relative_path
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite lock: {destination}")
    locked = {
        rel(root, path): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    }
    receipt = {
        "experiment": experiment,
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        **extras,
        "locked_files": locked,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def verify_file_lock(root: Path, relative_path: str, experiment: str) -> dict:
    destination = root / relative_path
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    if receipt.get("experiment") != experiment or receipt.get("revision") != 1:
        raise ValueError(f"unexpected lock identity at {destination}")
    for name, metadata in receipt.get("locked_files", {}).items():
        item = root / name
        if (not item.is_file() or item.stat().st_size != metadata["size_bytes"]
                or sha256(item) != metadata["sha256"]):
            raise ValueError(f"locked input changed: {name}")
    return receipt


def create_source_lock(root: Path) -> dict:
    destination = root / SOURCE_LOCK
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite source lock: {destination}")
    verify_ancestor_experiments(root)
    split_inputs, full_ids = split_files(root)
    return write_lock(
        root, SOURCE_LOCK, "Model406 component-recall plus hard-negative island suppression source",
        source_input_paths(root),
        {
            "metric_scope": "train-only grouped OOF; official valid/test excluded",
            "images": len(full_ids),
            "groups": 59,
            "folds": 4,
            "training_inputs": "outer-train symlink targets and fixed held-out OOF masks",
            "split_source_count": len(split_inputs),
            "gpu_training_started": False,
        },
    )


def verify_source_lock(root: Path) -> dict:
    receipt = verify_file_lock(
        root, SOURCE_LOCK,
        "Model406 component-recall plus hard-negative island suppression source",
    )
    if (receipt.get("images") != 70 or receipt.get("groups") != 59
            or receipt.get("folds") != 4 or receipt.get("metric_scope")
            != "train-only grouped OOF; official valid/test excluded"):
        raise ValueError("Model406 source lock has an unexpected evaluation scope")
    expected = {rel(root, path) for path in source_input_paths(root)}
    if expected != set(receipt.get("locked_files", {})):
        raise ValueError("Model406 source input inventory changed")
    verify_ancestor_experiments(root)
    return receipt


def source_lock_sha(root: Path) -> str:
    path = root / SOURCE_LOCK
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != (
        "Model406 component-recall plus hard-negative island suppression source"
    ):
        raise ValueError("Model406 source lock is missing or has an unexpected identity")
    return sha256(path)


def validate_calibration(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment")
            != "Model406 component-recall plus hard-negative island loss calibration"
            or record.get("valid_read") is not False
            or record.get("test_read") is not False
            or record.get("optimizer_steps") != 0
            or set(record.get("folds", {})) != {"1", "2", "3", "4"}):
        raise ValueError("Model406 calibration is missing or out of scope")
    for fold, item in record["folds"].items():
        weight = float(item["joint_aux_weight"])
        if (not item["training_split_only"] or item["optimizer_steps"] != 0
                or item["valid_read"] or item["test_read"]
                or item["nonempty_gradient_batches"] < 6
                or not 1e-5 <= weight <= 1.0):
            raise ValueError(f"unsafe Model406 calibration in fold {fold}")
    return record


def create_calibration_lock(root: Path) -> dict:
    source_hash = source_lock_sha(root)
    calibration_path = root / CALIBRATION
    calibration = validate_calibration(calibration_path)
    return write_lock(
        root, CALIBRATION_LOCK, "Model406 outer-train-only joint-loss calibration",
        [calibration_path],
        {
            "source_lock_sha256": source_hash,
            "valid_read": False,
            "test_read": False,
            "optimizer_steps": 0,
            "target_gradient_fraction": 0.10,
            "folds": sorted(calibration["folds"]),
        },
    )


def verify_calibration_lock(root: Path) -> dict:
    source_hash = source_lock_sha(root)
    receipt = verify_file_lock(
        root, CALIBRATION_LOCK, "Model406 outer-train-only joint-loss calibration"
    )
    if (receipt.get("source_lock_sha256") != source_hash
            or receipt.get("valid_read") is not False
            or receipt.get("test_read") is not False
            or receipt.get("optimizer_steps") != 0
            or receipt.get("folds") != ["1", "2", "3", "4"]):
        raise ValueError("Model406 calibration lock is invalid")
    validate_calibration(root / CALIBRATION)
    return receipt


def oof_output_paths(root: Path) -> list[Path]:
    verify_calibration_lock(root)
    masks_by_fold = m396.mask_files(root)
    output_root = root / OUTPUT_ROOT
    files: list[Path] = []
    calibration = validate_calibration(root / CALIBRATION)
    for fold, mask_map in masks_by_fold.items():
        expected_ids = set(mask_map)
        parent = root / MODEL392_ROOT / f"fold{fold}/last.pt"
        for arm in ("control", "candidate"):
            run = output_root / arm / f"fold{fold}"
            oof = output_root / arm / f"fold{fold}_oof"
            args_path = run / "args.json"
            checkpoint = run / "last.pt"
            probability_path = oof / "valid_probabilities.npz"
            summary_path = oof / "valid_summary.json"
            per_image_path = oof / "valid_per_image.csv"
            for path in (args_path, checkpoint, probability_path, summary_path, per_image_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            run_args = json.loads(args_path.read_text(encoding="utf-8"))
            if (run_args.get("arm") != arm or run_args.get("epochs") != 8
                    or run_args.get("seed") != 1
                    or Path(run_args.get("init_checkpoint", "")).resolve() != parent.resolve()
                    or Path(run_args.get("data_root", "")).resolve()
                    != (root / SPLIT_ROOT / f"seed1_fold{fold}").resolve()):
                raise ValueError(f"Model406 training recipe/split mismatch: {arm}/fold{fold}")
            expected_weight = (
                float(calibration["folds"][str(fold)]["joint_aux_weight"])
                if arm == "candidate" else 0.0
            )
            if not np.isclose(float(run_args.get("component_weight", -1.0)),
                              expected_weight, rtol=0.0, atol=1e-12):
                raise ValueError(f"Model406 auxiliary weight mismatch: {arm}/fold{fold}")
            with np.load(probability_path, allow_pickle=False) as archive:
                if set(archive.files) != expected_ids:
                    raise ValueError(f"Model406 OOF IDs mismatch: {arm}/fold{fold}")
            files.extend((args_path, checkpoint, probability_path, summary_path, per_image_path))
    return files


def create_oof_lock(root: Path) -> dict:
    source_hash = source_lock_sha(root)
    calibration = verify_calibration_lock(root)
    files = [root / CALIBRATION, root / CALIBRATION_LOCK, *oof_output_paths(root)]
    return write_lock(
        root, OOF_LOCK, "Model406 paired component/hard-negative grouped-OOF artifacts",
        files,
        {
            "source_lock_sha256": source_hash,
            "calibration_lock_sha256": sha256(root / CALIBRATION_LOCK),
            "metric_scope": "70 train images/59 specimen groups; official valid/test excluded",
            "official_valid_and_test_used": False,
            "gpu_training_used": True,
            "arms": ["control", "candidate"],
        },
    )


def verify_oof_lock(root: Path) -> dict:
    source_hash = source_lock_sha(root)
    calibration = verify_calibration_lock(root)
    receipt = verify_file_lock(
        root, OOF_LOCK, "Model406 paired component/hard-negative grouped-OOF artifacts"
    )
    if (receipt.get("source_lock_sha256") != source_hash
            or receipt.get("calibration_lock_sha256") != sha256(root / CALIBRATION_LOCK)
            or receipt.get("official_valid_and_test_used") is not False
            or receipt.get("metric_scope")
            != "70 train images/59 specimen groups; official valid/test excluded"):
        raise ValueError("Model406 OOF lock is invalid")
    expected = {
        rel(root, path) for path in
        [root / CALIBRATION, root / CALIBRATION_LOCK, *oof_output_paths(root)]
    }
    if expected != set(receipt.get("locked_files", {})):
        raise ValueError("Model406 OOF artifact inventory changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-source-lock", action="store_true")
    mode.add_argument("--verify-source-lock", action="store_true")
    mode.add_argument("--create-calibration-lock", action="store_true")
    mode.add_argument("--verify-calibration-lock", action="store_true")
    mode.add_argument("--create-oof-lock", action="store_true")
    mode.add_argument("--verify-oof-lock", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.create_source_lock:
        record = create_source_lock(root)
        path = root / SOURCE_LOCK
    elif args.verify_source_lock:
        record = verify_source_lock(root)
        path = root / SOURCE_LOCK
    elif args.create_calibration_lock:
        record = create_calibration_lock(root)
        path = root / CALIBRATION_LOCK
    elif args.verify_calibration_lock:
        record = verify_calibration_lock(root)
        path = root / CALIBRATION_LOCK
    elif args.create_oof_lock:
        record = create_oof_lock(root)
        path = root / OOF_LOCK
    else:
        record = verify_oof_lock(root)
        path = root / OOF_LOCK
    print(json.dumps({
        "lock": str(path),
        "locked_files": len(record.get("locked_files", {})),
        "sha256": sha256(path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
