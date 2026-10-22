#!/usr/bin/env python3
"""Fold-3/4 CPU attribution stop-gate; returns cancellation evidence, never GO."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids
from model383_boundary_relaxation_gate import relaxed_band


GROUP_RE = re.compile(r"^(A\d+)")
NUM_CLASSES = 4
FOLDS_FOR_DECISION = (3, 4)
EXPECTED_IMAGES_PER_FOLD = 17
BOOTSTRAP_SEED = 383
BOOTSTRAP_DRAWS = 20_000


def _image_miou(prediction: np.ndarray, truth: np.ndarray) -> float:
    # image_ious always returns all four classes and maps empty union to zero.
    return float(np.mean(image_ious(prediction, truth)))


def _fold_stratified_group_bootstrap(
    rows: list[dict[str, Any]], draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    groups_by_fold: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups_by_fold[row["fold"]][row["group"]].append(index)
    if set(groups_by_fold) != set(FOLDS_FOR_DECISION):
        raise ValueError("bootstrap requires exactly folds 3 and 4")

    base = np.asarray([r["f_empty"] for r in rows], dtype=np.float64)
    band_only = np.asarray([r["f_band"] for r in rows], dtype=np.float64)
    outside_only = np.asarray([r["f_outside"] for r in rows], dtype=np.float64)
    full = np.asarray([r["f_both"] for r in rows], dtype=np.float64)
    oracle = np.asarray([r["f_oracle"] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = np.empty((draws, 4), dtype=np.float64)

    for draw in range(draws):
        selected_indices: list[int] = []
        for fold in FOLDS_FOR_DECISION:
            group_map = groups_by_fold[fold]
            names = sorted(group_map)
            sampled = rng.integers(0, len(names), size=len(names))
            for position in sampled:
                selected_indices.extend(group_map[names[int(position)]])
        idx = np.asarray(selected_indices, dtype=np.int64)
        f0 = float(base[idx].mean())
        f_band = float(band_only[idx].mean())
        f_outside = float(outside_only[idx].mean())
        f_full = float(full[idx].mean())
        headroom = float((oracle[idx] - base[idx]).mean())
        delta_full = f_full - f0
        phi_band = 0.5 * ((f_band - f0) + (f_full - f_outside))
        ratio = phi_band / headroom if headroom > 0 else np.nan
        stats[draw] = (100 * delta_full, 100 * phi_band, 100 * headroom, ratio)

    def ci(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if finite.size < int(0.95 * draws):
            return None
        return np.percentile(finite, [2.5, 97.5]).astype(float).tolist()

    return {
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
        "stratification": "resample specimen groups with replacement within folds 3 and 4; carry all images in each sampled group",
        "model376_full_delta_pp_95ci": ci(stats[:, 0]),
        "model376_shapley_band_contribution_pp_95ci": ci(stats[:, 1]),
        "model374_band_oracle_headroom_pp_95ci": ci(stats[:, 2]),
        "shapley_band_over_headroom_ratio_95ci": ci(stats[:, 3]),
    }


def analyze(cv_root: Path, model374_root: Path, model376_root: Path,
            draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
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
                    if probs.min() < -1e-4 or probs.max() > 1.0001:
                        raise ValueError(f"out-of-range Model{name} probabilities for {image_id}")
                    if np.max(np.abs(probs.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized Model{name} probabilities for {image_id}")

                pred374 = probs374.argmax(axis=0).astype(np.uint8)
                pred376 = probs376.argmax(axis=0).astype(np.uint8)
                band = relaxed_band(truth)
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
                    "f_empty": _image_miou(pred374, truth),
                    "f_band": _image_miou(replace_in_band, truth),
                    "f_outside": _image_miou(replace_outside_band, truth),
                    "f_both": _image_miou(pred376, truth),
                    "f_oracle": _image_miou(oracle, truth),
                    "wrong_pixels": int((pred374 != truth).sum()),
                    "wrong_pixels_in_band": int(((pred374 != truth) & band).sum()),
                    "wrong_pixels_outside_band": int(((pred374 != truth) & ~band).sum()),
                })

    if len(rows) != 2 * EXPECTED_IMAGES_PER_FOLD:
        raise ValueError(f"expected 34 decision images, found {len(rows)}")
    base = np.asarray([r["f_empty"] for r in rows], dtype=np.float64)
    band_only = np.asarray([r["f_band"] for r in rows], dtype=np.float64)
    outside_only = np.asarray([r["f_outside"] for r in rows], dtype=np.float64)
    full = np.asarray([r["f_both"] for r in rows], dtype=np.float64)
    oracle = np.asarray([r["f_oracle"] for r in rows], dtype=np.float64)
    delta_full_pp = float(100 * (full - base).mean())
    phi_band_pp = float(50 * ((band_only - base) + (full - outside_only)).mean())
    phi_outside_pp = float(100 * (full - base).mean() - phi_band_pp)
    headroom_pp = float(100 * (oracle - base).mean())
    ratio_point = phi_band_pp / headroom_pp if headroom_pp > 0 else None
    total_wrong = sum(r["wrong_pixels"] for r in rows)
    inside_wrong = sum(r["wrong_pixels_in_band"] for r in rows)
    bootstrap = _fold_stratified_group_bootstrap(rows, draws=draws)
    delta_ci = bootstrap["model376_full_delta_pp_95ci"]
    ratio_ci = bootstrap["shapley_band_over_headroom_ratio_95ci"]
    cancel = bool(delta_ci is not None and ratio_ci is not None and delta_ci[0] > 0.0 and ratio_ci[1] < 0.03)

    return {
        "model": 383,
        "purpose": "proxy cancellation test only; not a GO gate",
        "scope": "train-only grouped OOF folds 3-4; selection-conditional",
        "metric": "image-wise four-class mIoU; empty-union=0; fixed four-class mean per image",
        "images": len(rows),
        "groups": len(seen_groups),
        "folds": list(FOLDS_FOR_DECISION),
        "model374_base_miou_pct": float(base.mean() * 100),
        "model376_full_miou_pct": float(full.mean() * 100),
        "model376_full_delta_vs374_pp": delta_full_pp,
        "model376_shapley_band_contribution_pp": phi_band_pp,
        "model376_shapley_outside_contribution_pp": phi_outside_pp,
        "model376_shapley_sum_equals_full_delta_pp": float(phi_band_pp + phi_outside_pp),
        "model374_band_oracle_headroom_pp_descriptive_only": headroom_pp,
        "shapley_band_over_headroom_ratio_point_estimate": ratio_point,
        "wrong_pixel_fraction_inside_band": float(inside_wrong / total_wrong) if total_wrong else 0.0,
        "wrong_pixel_fraction_outside_band": float((total_wrong - inside_wrong) / total_wrong) if total_wrong else 0.0,
        "bootstrap": bootstrap,
        "proxy_cancellation_rule": "cancel only if 95% group-bootstrap CI lower for Model376 full delta > 0 AND CI upper for Shapley-band/headroom ratio < 0.03",
        "cancel_model383_for_proxy_evidence": cancel,
        "otherwise": "inconclusive; this result alone does not authorize Model383 training",
        "source_review_required": "Check Model376 implementation only for exact duplication; nonidentity alone is not evidence BLR will work.",
        "per_fold": {
            str(fold): {
                "images": sum(r["fold"] == fold for r in rows),
                "model374_base_miou_pct": float(np.mean([r["f_empty"] for r in rows if r["fold"] == fold]) * 100),
                "model376_full_miou_pct": float(np.mean([r["f_both"] for r in rows if r["fold"] == fold]) * 100),
                "model376_shapley_band_contribution_pp": float(50 * np.mean([
                    (r["f_band"] - r["f_empty"]) + (r["f_both"] - r["f_outside"])
                    for r in rows if r["fold"] == fold
                ])),
            }
            for fold in FOLDS_FOR_DECISION
        },
        "per_image": rows,
        "interpretation_guard": "A model prediction attribution proxy cannot prove the counterfactual effect of a different training loss. Historical method selection is not nested.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--model376-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.draws < 100:
        raise SystemExit("--draws must be at least 100; registered run uses 20000")
    result = analyze(args.cv_root, args.model374_root, args.model376_root, draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
