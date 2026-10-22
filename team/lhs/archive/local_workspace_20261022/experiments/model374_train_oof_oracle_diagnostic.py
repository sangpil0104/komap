#!/usr/bin/env python3
"""Locate correctable Model374 errors using training-only grouped OOF data.

The two analyses use ground truth and are diagnostic interventions, never
deployable inference methods or validation scores. Their mIoU can decrease.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from model374_train_oof_error_diagnostic import (
    CLASS_NAMES,
    NUM_CLASSES,
    image_ious,
    mask_to_ids,
)


def score(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(image_ious(pred, target)))


def presence_oracle(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    present = np.bincount(target.ravel(), minlength=NUM_CLASSES) > 0
    allowed = np.where(present[:, None, None], probabilities, -np.inf)
    return allowed.argmax(axis=0).astype(np.uint8)


def component_label_oracle(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int, dict]:
    """Assign each 8-connected predicted component its GT-majority phase."""
    corrected = pred.copy()
    relabeled = 0
    transition_components = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    transition_pixels = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    changed_size_bins = np.zeros(5, dtype=np.int64)
    for class_id in range(NUM_CLASSES):
        region = (pred == class_id).astype(np.uint8)
        count, labels = cv2.connectedComponents(region, connectivity=8)
        if count <= 1:
            continue
        component_ids = labels[region != 0].astype(np.int64)
        truth_ids = target[region != 0].astype(np.int64)
        votes = np.bincount(
            component_ids * NUM_CLASSES + truth_ids,
            minlength=count * NUM_CLASSES,
        ).reshape(count, NUM_CLASSES)
        chosen = votes.argmax(axis=1).astype(np.uint8)
        # An equal-vote result has no evidence to change the predicted phase.
        chosen[votes[np.arange(count), class_id] == votes.max(axis=1)] = class_id
        changed = chosen[1:] != class_id
        relabeled += int(np.count_nonzero(changed))
        sizes = votes.sum(axis=1)
        destinations = chosen[1:][changed].astype(np.int64)
        changed_sizes = sizes[1:][changed]
        np.add.at(transition_components[class_id], destinations, 1)
        np.add.at(transition_pixels[class_id], destinations, changed_sizes)
        changed_size_bins += np.histogram(changed_sizes, bins=[0, 4, 16, 64, 256, np.inf])[0]
        corrected[region != 0] = chosen[component_ids]
    details = {
        "transition_components": transition_components.tolist(),
        "transition_pixels": transition_pixels.tolist(),
        "changed_size_bins_lt4_lt16_lt64_lt256_ge256": changed_size_bins.tolist(),
        "pixels_relabelled": int(np.count_nonzero(corrected != pred)),
    }
    return corrected, relabeled, details


def analyze(cv_data: Path, oof_root: Path, folds: tuple[int, ...]) -> dict:
    rows = []
    seen = set()
    for fold in folds:
        mask_dir = cv_data / f"oof_fold{fold}" / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"missing fold {fold} masks or probabilities")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold} prediction and mask IDs differ")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image: {image_id}")
                seen.add(image_id)
                target = mask_to_ids(masks[image_id])
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                if probabilities.shape != (NUM_CLASSES, *target.shape):
                    raise ValueError(f"bad probability shape for {image_id}: {probabilities.shape}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"non-finite probabilities for {image_id}")
                pred = probabilities.argmax(axis=0).astype(np.uint8)
                presence = presence_oracle(probabilities, target)
                component, changed_components, details = component_label_oracle(pred, target)
                true_present = np.bincount(target.ravel(), minlength=NUM_CLASSES) > 0
                pred_present = np.bincount(pred.ravel(), minlength=NUM_CLASSES) > 0
                rows.append({
                    "image_id": image_id,
                    "fold": fold,
                    "base_image_miou_empty0": score(pred, target),
                    "perfect_presence_image_miou_empty0": score(presence, target),
                    "perfect_component_labels_image_miou_empty0": score(component, target),
                    "predicted_absent_classes": [CLASS_NAMES[i] for i in range(NUM_CLASSES)
                                                 if pred_present[i] and not true_present[i]],
                    "missed_present_classes": [CLASS_NAMES[i] for i in range(NUM_CLASSES)
                                               if true_present[i] and not pred_present[i]],
                    "components_relabelled_by_oracle": changed_components,
                    **details,
                })
    if not rows:
        raise ValueError("no grouped OOF images")
    base = np.asarray([row["base_image_miou_empty0"] for row in rows])
    presence = np.asarray([row["perfect_presence_image_miou_empty0"] for row in rows])
    component = np.asarray([row["perfect_component_labels_image_miou_empty0"] for row in rows])
    transition_components = np.sum([row["transition_components"] for row in rows], axis=0)
    transition_pixels = np.sum([row["transition_pixels"] for row in rows], axis=0)
    size_bins = np.sum([row["changed_size_bins_lt4_lt16_lt64_lt256_ge256"] for row in rows], axis=0)
    return {
        "scope": "train-only grouped OOF; no official validation or test data",
        "metric": "image mean of four class IoUs; empty union=0",
        "folds": list(folds),
        "images": len(rows),
        "base_image_mean_miou_empty0": float(base.mean()),
        "perfect_presence_image_mean_miou_empty0": float(presence.mean()),
        "perfect_presence_gain_percentage_points": float(100 * (presence - base).mean()),
        "perfect_component_labels_image_mean_miou_empty0": float(component.mean()),
        "perfect_component_labels_gain_percentage_points": float(100 * (component - base).mean()),
        "images_with_false_present_phase": sum(bool(row["predicted_absent_classes"]) for row in rows),
        "images_with_missed_phase": sum(bool(row["missed_present_classes"]) for row in rows),
        "oracle_component_transition_components_from_row_to_column": transition_components.tolist(),
        "oracle_component_transition_pixels_from_row_to_column": transition_pixels.tolist(),
        "class_order": CLASS_NAMES,
        "oracle_changed_component_size_bins_lt4_lt16_lt64_lt256_ge256": size_bins.tolist(),
        "oracle_pixels_relabelled": sum(row["pixels_relabelled"] for row in rows),
        "interpretation_guard": (
            "Both interventions inspect ground truth; their gains cannot be claimed as model gains. "
            "They may reduce mIoU and are not mathematical upper bounds."
        ),
        "per_image": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = tuple(args.folds)
    if not folds or len(set(folds)) != len(folds) or any(fold not in (1, 2, 3, 4) for fold in folds):
        raise SystemExit("--folds must contain unique fold numbers from 1 to 4")
    result = analyze(args.cv_data, args.oof_root, folds)
    if result["images"] != args.expected_images:
        raise SystemExit(f"expected {args.expected_images} OOF images; got {result['images']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
