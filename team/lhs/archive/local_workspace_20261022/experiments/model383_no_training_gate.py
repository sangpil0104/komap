#!/usr/bin/env python3
"""Fold-3/4 CPU stop-gate for Model383; requires aligned Model374/376 OOF maps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids
from model383_boundary_relaxation_gate import euclidean_distance_to_band, relaxed_band


GROUP_RE = re.compile(r"^(A\d+)")
NUM_CLASSES = 4
FOLDS_FOR_DECISION = (3, 4)
EXPECTED_IMAGES_PER_FOLD = 17


def _image_miou(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(image_ious(prediction, truth)))


def analyze(cv_root: Path, model374_root: Path, model376_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_groups: dict[str, int] = {}
    for fold in FOLDS_FOR_DECISION:
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        path374 = model374_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        path376 = model376_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not path374.is_file() or not path376.is_file():
            raise FileNotFoundError(f"fold {fold}: missing GT masks or Model374/376 OOF archive")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if len(masks) != EXPECTED_IMAGES_PER_FOLD:
            raise ValueError(f"fold {fold}: expected 17 masks, found {len(masks)}")
        with np.load(path374, allow_pickle=False) as archive374, np.load(path376, allow_pickle=False) as archive376:
            if set(archive374.files) != set(masks) or set(archive376.files) != set(masks):
                raise ValueError(f"fold {fold}: OOF probability IDs do not exactly match masks")
            for image_id in sorted(masks):
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group in {image_id}")
                group = match.group(1)
                if group in seen_groups:
                    raise ValueError(f"specimen group {group} occurs in more than one decision fold")
                seen_groups[group] = fold

                truth = mask_to_ids(masks[image_id])
                probs374 = np.asarray(archive374[image_id], dtype=np.float32)
                probs376 = np.asarray(archive376[image_id], dtype=np.float32)
                expected_shape = (NUM_CLASSES, *truth.shape)
                for name, probs in (("374", probs374), ("376", probs376)):
                    if probs.shape != expected_shape or not np.isfinite(probs).all():
                        raise ValueError(f"invalid Model{name} probabilities for {image_id}")
                    if np.max(np.abs(probs.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized Model{name} probabilities for {image_id}")

                pred374 = probs374.argmax(axis=0).astype(np.uint8)
                pred376 = probs376.argmax(axis=0).astype(np.uint8)
                band = relaxed_band(truth)
                distance = euclidean_distance_to_band(band)
                wrong = pred374 != truth
                band_wrong = wrong & band
                gt_prob = np.take_along_axis(probs374, truth[None], axis=0)[0]
                top1_prob = probs374.max(axis=0)
                gt_rank = 1 + (probs374 > gt_prob[None]).sum(axis=0)
                low_margin_top2 = band_wrong & (gt_rank <= 2) & ((top1_prob - gt_prob) < 0.20)

                oracle = pred374.copy()
                oracle[band] = truth[band]
                replace_in_band = pred374.copy()
                replace_in_band[band] = pred376[band]
                replace_outside_band = pred376.copy()
                replace_outside_band[band] = pred374[band]

                rows.append({
                    "image_id": image_id,
                    "group": group,
                    "fold": fold,
                    "base374_miou": _image_miou(pred374, truth),
                    "full376_miou": _image_miou(pred376, truth),
                    "oracle374_band_miou": _image_miou(oracle, truth),
                    "replace374_with376_inside_band_miou": _image_miou(replace_in_band, truth),
                    "replace374_with376_outside_band_miou": _image_miou(replace_outside_band, truth),
                    "band_error_pixels": int(band_wrong.sum()),
                    "band_error_pixels_gt_top2_margin_lt_0_20": int(low_margin_top2.sum()),
                    "wrong_pixels": int(wrong.sum()),
                    "wrong_pixels_gt3px_from_band": int((wrong & (distance > 3.0)).sum()),
                    "relaxed_band_fraction": float(band.mean()),
                })

    base = np.asarray([r["base374_miou"] for r in rows], dtype=np.float64)
    full376 = np.asarray([r["full376_miou"] for r in rows], dtype=np.float64)
    oracle = np.asarray([r["oracle374_band_miou"] for r in rows], dtype=np.float64)
    band_swap = np.asarray([r["replace374_with376_inside_band_miou"] for r in rows], dtype=np.float64)
    outside_swap = np.asarray([r["replace374_with376_outside_band_miou"] for r in rows], dtype=np.float64)
    oracle_gain_pp = float(100 * (oracle - base).mean())
    model376_total_delta_pp = float(100 * (full376 - base).mean())
    model376_band_swap_delta_pp = float(100 * (band_swap - base).mean())
    model376_outside_swap_delta_pp = float(100 * (outside_swap - base).mean())
    total_wrong = sum(r["wrong_pixels"] for r in rows)
    far_fraction = sum(r["wrong_pixels_gt3px_from_band"] for r in rows) / max(total_wrong, 1)
    band_errors = sum(r["band_error_pixels"] for r in rows)
    low_margin_fraction = sum(r["band_error_pixels_gt_top2_margin_lt_0_20"] for r in rows) / max(band_errors, 1)
    model376_mostly_band = model376_total_delta_pp > 0 and model376_band_swap_delta_pp >= 0.5 * model376_total_delta_pp
    model376_low_headroom_recovery = model376_band_swap_delta_pp < 0.03 * oracle_gain_pp
    gate_values = {
        "fold34_band_oracle_gain_at_least_5pp": oracle_gain_pp >= 5.0,
        "fold34_errors_more_than_3px_from_band_below_15pct": far_fraction < 0.15,
        "fold34_band_errors_top2_margin_lt_0_20_at_least_30pct": low_margin_fraction >= 0.30,
        "model376_does_not_show_low_band_headroom_recovery": not (model376_mostly_band and model376_low_headroom_recovery),
    }
    return {
        "model": 383,
        "purpose": "no-training stop gate; folds 3-4 are the decision set, folds 1-2 are excluded",
        "scope": "train-only grouped OOF; selection-conditional",
        "metric": "image-wise four-class mIoU; empty-union=0",
        "images": len(rows),
        "groups": len(seen_groups),
        "folds": list(FOLDS_FOR_DECISION),
        "model374_base_miou_pct": float(base.mean() * 100),
        "model374_relaxed_band_oracle_miou_pct": float(oracle.mean() * 100),
        "model374_relaxed_band_oracle_gain_pp": oracle_gain_pp,
        "wrong_pixel_fraction_gt3px_from_relaxed_band": float(far_fraction),
        "band_error_fraction_gt_top2_and_margin_lt_0_20": float(low_margin_fraction),
        "model376_full_delta_vs374_pp": model376_total_delta_pp,
        "model376_band_only_replacement_delta_vs374_pp": model376_band_swap_delta_pp,
        "model376_outside_only_replacement_delta_vs374_pp": model376_outside_swap_delta_pp,
        "model376_band_share_of_positive_total_gain": (
            float(model376_band_swap_delta_pp / model376_total_delta_pp)
            if model376_total_delta_pp > 0 else None
        ),
        "model376_band_headroom_recovery_fraction": (
            float(model376_band_swap_delta_pp / oracle_gain_pp)
            if oracle_gain_pp != 0 else None
        ),
        "model376_mostly_band": bool(model376_mostly_band),
        "model376_low_headroom_recovery": bool(model376_low_headroom_recovery),
        "no_training_gates": gate_values,
        "go_to_model383_source_review": all(gate_values.values()),
        "source_review_required": "Model376 loss implementation must be recovered and shown not to already implement the same 3x3 neighborhood-union likelihood.",
        "per_fold": {
            str(fold): {
                "images": sum(r["fold"] == fold for r in rows),
                "model374_base_miou_pct": float(np.mean([r["base374_miou"] for r in rows if r["fold"] == fold]) * 100),
                "model376_full_miou_pct": float(np.mean([r["full376_miou"] for r in rows if r["fold"] == fold]) * 100),
                "model374_band_oracle_miou_pct": float(np.mean([r["oracle374_band_miou"] for r in rows if r["fold"] == fold]) * 100),
            }
            for fold in FOLDS_FOR_DECISION
        },
        "per_image": rows,
        "interpretation_guard": "Failing or missing any prerequisite cancels Model383. These gates do not make the historical method-selection process nested or unbiased.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--model376-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.cv_root, args.model374_root, args.model376_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
