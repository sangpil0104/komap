#!/usr/bin/env python3
"""Exploratory class FN/FP diagnostics on fixed grouped train-OOF predictions.

Reads only the four fixed ``oof_foldN/masks`` training holdouts and the B4 / two
Model392 probability archives already locked for Model397. It reports confusion
and per-image error rates, not mIoU, and never opens official valid/test inputs.
Results are post-selection, exploratory diagnostics, not independent evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import model396_fixed_architecture_oof as model396
import model397_fixed_b4_model392_blend as model397


ROOT = Path(__file__).resolve().parents[1]
RESULT_REL = "experiments/results/model397_train_oof_error_type_audit.json"
CLASS_NAMES = model396.CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("truth and prediction must be same-size 2-D class maps")
    if int(truth.min()) < 0 or int(truth.max()) >= NUM_CLASSES:
        raise ValueError("truth class IDs outside fixed palette")
    if int(prediction.min()) < 0 or int(prediction.max()) >= NUM_CLASSES:
        raise ValueError("predicted class IDs outside fixed palette")
    return np.bincount(
        truth.astype(np.int64).ravel() * NUM_CLASSES + prediction.astype(np.int64).ravel(),
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def _check_probabilities(probabilities: np.ndarray, truth: np.ndarray, image_id: str) -> None:
    if probabilities.shape != (NUM_CLASSES, *truth.shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values for {image_id}: {probabilities.shape}")
    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError(f"probability outside [0,1] for {image_id}")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError(f"probabilities are not normalized for {image_id}")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty diagnostic subset")
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for row in rows:
        confusion += np.asarray(row["confusion"], dtype=np.int64)
    total_pixels = int(confusion.sum())
    classes: dict[str, Any] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        gt_areas: list[int] = []
        pred_areas: list[int] = []
        true_positives: list[int] = []
        false_positives: list[int] = []
        false_negatives: list[int] = []
        image_pixels: list[int] = []
        for row in rows:
            matrix = np.asarray(row["confusion"], dtype=np.int64)
            gt = int(matrix[class_id, :].sum())
            predicted = int(matrix[:, class_id].sum())
            tp = int(matrix[class_id, class_id])
            gt_areas.append(gt)
            pred_areas.append(predicted)
            true_positives.append(tp)
            false_positives.append(predicted - tp)
            false_negatives.append(gt - tp)
            image_pixels.append(int(matrix.sum()))

        present = [i for i, gt in enumerate(gt_areas) if gt > 0]
        absent = [i for i, gt in enumerate(gt_areas) if gt == 0]
        pooled_gt = int(confusion[class_id, :].sum())
        pooled_pred = int(confusion[:, class_id].sum())
        pooled_tp = int(confusion[class_id, class_id])
        classes[name] = {
            "pooled_pixels": {
                "tp": pooled_tp,
                "fp": pooled_pred - pooled_tp,
                "fn": pooled_gt - pooled_tp,
                "gt": pooled_gt,
                "predicted": pooled_pred,
                "recall": pooled_tp / pooled_gt if pooled_gt else None,
                "precision": pooled_tp / pooled_pred if pooled_pred else None,
            },
            "gt_present_images": len(present),
            "gt_absent_images": len(absent),
            "gt_present_image_means": {
                "recall": float(np.mean([true_positives[i] / gt_areas[i] for i in present])) if present else None,
                "precision_zero_if_no_predictions": float(np.mean([
                    true_positives[i] / pred_areas[i] if pred_areas[i] else 0.0 for i in present
                ])) if present else None,
                "gt_area_percent": float(np.mean([
                    100.0 * gt_areas[i] / image_pixels[i] for i in present
                ])) if present else None,
                "predicted_area_percent": float(np.mean([
                    100.0 * pred_areas[i] / image_pixels[i] for i in present
                ])) if present else None,
                "false_positive_image_area_percent": float(np.mean([
                    100.0 * false_positives[i] / image_pixels[i] for i in present
                ])) if present else None,
                "false_negative_image_area_percent": float(np.mean([
                    100.0 * false_negatives[i] / image_pixels[i] for i in present
                ])) if present else None,
                "predicted_to_gt_area_ratio": float(np.mean([
                    pred_areas[i] / gt_areas[i] for i in present
                ])) if present else None,
            },
            "gt_absent_image_hallucination": {
                "images_with_any_prediction": sum(pred_areas[i] > 0 for i in absent),
                "fraction_images_with_any_prediction": float(np.mean([
                    pred_areas[i] > 0 for i in absent
                ])) if absent else None,
                "mean_predicted_area_percent": float(np.mean([
                    100.0 * pred_areas[i] / image_pixels[i] for i in absent
                ])) if absent else None,
            },
        }
    return {
        "images": len(rows),
        "specimen_groups": len({row["group"] for row in rows}),
        "pooled_confusion_rows_truth_columns_prediction": confusion.tolist(),
        "total_pixels": total_pixels,
        "classes": classes,
    }


def _verify_locked_file(root: Path, path: Path, lock: dict[str, Any]) -> None:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    metadata = lock.get("locked_files", {}).get(relative)
    if metadata is None:
        raise ValueError(f"OOF diagnostic input is not in the fixed Model397 lock: {relative}")
    if path.stat().st_size != metadata["size_bytes"] or sha256(path) != metadata["sha256"]:
        raise ValueError(f"fixed Model397 input changed: {relative}")


def _verify_locked_oof_inputs(root: Path, fold: int, image_ids: set[str],
                              seed: int, path: Path, lock: dict[str, Any]) -> None:
    _verify_locked_file(root, path, lock)
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    metadata = lock["locked_files"][relative]
    if path.name == "valid_probabilities.npz":
        manifest_path = path.parent / "model392_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("model") != 392 or manifest.get("seed") != seed
                or manifest.get("fold") != fold
                or manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
                or manifest.get("oof_probabilities_sha256") != metadata["sha256"]):
            raise ValueError(f"Model392 held-out train-OOF manifest mismatch: {manifest_path}")
        data_root = root / f"data_model382_seed{seed}_fold{fold}"
        if (data_root / "valid").resolve() != (root / "data_model322_group_cv" / f"oof_fold{fold}").resolve():
            raise ValueError(f"Model392 archive is not linked to fixed train OOF fold {fold}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != image_ids:
            raise ValueError(f"OOF archive IDs mismatch: {relative}")


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    lock_path = root / model397.LOCK
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"
            or lock.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or lock.get("valid_and_test_inputs_excluded") is not True):
        raise ValueError("unexpected Model397 OOF-only lock")
    lock_hash = sha256(lock_path)
    oof_result = json.loads((root / model397.RESULT).read_text(encoding="utf-8"))
    if (oof_result.get("lock_sha256") != lock_hash
            or oof_result.get("official_valid_used") is not False
            or oof_result.get("test_used") is not False
            or oof_result.get("valid_stage_allowed") is not False):
        raise ValueError("Model397 result is not the locked train-OOF-only result")

    masks = model396.mask_files(root)
    b4_root = root / model397.B4_ROOT
    model392_root = root / model397.MODEL392_ROOT
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("B4", "H", "C")}
    fold_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {str(fold): [] for fold in range(1, 5)} for name in rows
    }
    used_hashes: dict[str, str] = {}

    for fold, fold_masks in masks.items():
        ids = set(fold_masks)
        b4_path = b4_root / f"fold{fold}_oof/probabilities.npz"
        b4_args_path = b4_root / f"fold{fold}/args.json"
        _verify_locked_file(root, b4_args_path, lock)
        b4_args = json.loads(b4_args_path.read_text(encoding="utf-8"))
        if (any(b4_args.get(key) != value for key, value in model396.B4_EXPECTED_ARGS.items())
                or b4_args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 input is not the fixed grouped train-OOF fold {fold}")
        _verify_locked_oof_inputs(root, fold, ids, 0, b4_path, lock)
        used_hashes[b4_path.resolve().relative_to(root).as_posix()] = sha256(b4_path)

        seed_paths = {
            seed: model392_root / f"seed{seed}/fold{fold}_oof/valid_probabilities.npz"
            for seed in (1, 2)
        }
        for seed, path in seed_paths.items():
            _verify_locked_oof_inputs(root, fold, ids, seed, path, lock)
            used_hashes[path.resolve().relative_to(root).as_posix()] = sha256(path)

        for image_id, mask_path in fold_masks.items():
            mask_relative = mask_path.resolve().relative_to(root).as_posix()
            mask_metadata = lock.get("locked_files", {}).get(mask_relative)
            if (mask_metadata is None or mask_path.stat().st_size != mask_metadata["size_bytes"]
                    or sha256(mask_path) != mask_metadata["sha256"]):
                raise ValueError(f"fixed train-OOF mask changed or not locked: {mask_relative}")
            used_hashes[mask_relative] = sha256(mask_path)

        with np.load(b4_path, allow_pickle=False) as b4_archive, \
             np.load(seed_paths[1], allow_pickle=False) as seed1_archive, \
             np.load(seed_paths[2], allow_pickle=False) as seed2_archive:
            for image_id, mask_path in sorted(fold_masks.items()):
                truth = model396.mask_to_ids(mask_path)
                p_b4 = np.asarray(b4_archive[image_id], dtype=np.float32)
                p_s1 = np.asarray(seed1_archive[image_id], dtype=np.float32)
                p_s2 = np.asarray(seed2_archive[image_id], dtype=np.float32)
                _check_probabilities(p_b4, truth, image_id)
                _check_probabilities(p_s1, truth, image_id)
                _check_probabilities(p_s2, truth, image_id)
                p_b4, p_h, p_c = model397.blend_probabilities(p_b4, p_s1, p_s2)
                group_match = model397.GROUP_RE.match(image_id)
                step_match = model397.STEP_RE.search(image_id)
                if group_match is None or step_match is None:
                    raise ValueError(f"invalid grouped OOF image ID: {image_id}")
                for name, probability in (("B4", p_b4), ("H", p_h), ("C", p_c)):
                    prediction = probability.argmax(axis=0).astype(np.uint8)
                    row = {
                        "group": group_match.group(1),
                        "step": int(step_match.group(1)),
                        "confusion": _confusion(truth, prediction).tolist(),
                    }
                    rows[name].append(row)
                    fold_rows[name][str(fold)].append(row)

    if any(len(items) != 70 for items in rows.values()):
        raise ValueError("expected exactly-once 70-image train OOF coverage")
    return {
        "analysis": "exploratory class false-negative/false-positive audit on Model397 fixed train OOF",
        "scope": "70 held-out training images; 59 specimen groups; four fixed grouped folds",
        "class_order": list(CLASS_NAMES),
        "arms": {
            "B4": "fixed Model329 fold-held-out probability",
            "H": "0.5 * Model392 seed1 + 0.5 * Model392 seed2",
            "C": "0.5 * Model329 B4 + 0.25 * Model392 seed1 + 0.25 * Model392 seed2",
        },
        "diagnostic_only": True,
        "official_miou_scored": False,
        "official_valid_or_test_read": False,
        "training_started": False,
        "gpu_used": False,
        "post_selection_exploratory": True,
        "input_sha256": used_hashes,
        "arms_summary": {
            name: {
                "all_oof_images": _summarize(rows[name]),
                "step4_images": _summarize([row for row in rows[name] if row["step"] == 4]),
                "by_fold": {
                    fold: _summarize(fold_rows[name][fold])
                    for fold in sorted(fold_rows[name], key=int)
                },
            }
            for name in rows
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_REL)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "analysis": result["analysis"],
        "images": result["arms_summary"]["C"]["all_oof_images"]["images"],
        "groups": result["arms_summary"]["C"]["all_oof_images"]["specimen_groups"],
        "official_miou_scored": result["official_miou_scored"],
        "official_valid_or_test_read": result["official_valid_or_test_read"],
        "training_started": result["training_started"],
        "gpu_used": result["gpu_used"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
