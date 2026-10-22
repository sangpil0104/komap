#!/usr/bin/env python3
"""Oracle boundary-tolerance audit for the fixed Model360 valid predictions.

This is diagnostic only: GT is used to identify and correct errors within a
multiclass boundary band. It does not describe a deployable postprocessor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import model355_model349_spatial_audit as spatial
from model360_entropy_gated_d4_into_355 import entropy

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def boundary_band(target: np.ndarray, radius: int) -> np.ndarray:
    edge = np.zeros(target.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        changed = target[tuple(left)] != target[tuple(right)]
        edge[tuple(left)] |= changed
        edge[tuple(right)] |= changed
    return ndimage.binary_dilation(edge, structure=STRUCTURE_8, iterations=radius)


def metric(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    classes = spatial.audit.class_iou(pred, target)
    return float(classes.mean()), classes


def repair(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, int]:
    errors = pred != target
    edited = errors & region
    corrected = pred.copy()
    corrected[edited] = target[edited]
    return corrected, int(edited.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=(1, 3))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    outputs = args.root / "outputs"
    rows, names, base_route, gamma329, _ = spatial.audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    load_archive = spatial.audit.model350.model341.route.load_archive
    archives = {}
    for key, path in (
        ("347", outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"),
        ("351", outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"),
    ):
        _, archives[key] = load_archive(path, team_order=True)
        if set(archives[key]) != set(names):
            raise ValueError(f"Model{key} validation image keys differ")
    model346 = json.loads((outputs / "model346_model336_into_341_route/results.json").read_text())
    model349 = json.loads((outputs / "model349_model347_into_346_route/results.json").read_text())
    model355 = json.loads((outputs / "model355_model349_spatial_audit/results.json").read_text())
    model360 = json.loads((outputs / "model360_entropy_gated_d4_into_355/results.json").read_text())
    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    blend_primary, blend_eutectic = model355["raw_best_spatial_blend"]
    margin, strength = model360["raw_best_entropy_gate"]
    fixed_route = dict(base_route, gaussian_blend=blend_primary,
                       eut_edge_blend=blend_eutectic)

    per_image = []
    for index, (name, row) in enumerate(zip(names, rows)):
        source336 = row["sources"]["336"]
        source351 = archives["351"][name]
        gate = (entropy(source351) + margin < entropy(source336)).astype(np.float32)
        row["sources"]["336"] = source336 + strength * gate[None] * (source351 - source336)
        row["sources"]["347"] = archives["347"][name]
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        pred = spatial.prediction(parts, fixed_route, blend_primary, blend_eutectic)
        row["sources"]["336"] = source336
        target = row["item"]["target"].astype(np.uint8)
        base_score, base_classes = metric(pred, target)
        record = {
            "image": name,
            "baseline_image_miou_empty0": base_score,
            "baseline_class_iou": dict(zip(spatial.audit.model350.CLASS_ORDER,
                                             base_classes.tolist())),
            "pixels": int(target.size),
            "misclassified_pixels": int((pred != target).sum()),
        }
        for radius in args.radii:
            band = boundary_band(target, radius)
            fixed_near, corrected_near = repair(pred, target, band)
            fixed_far, corrected_far = repair(pred, target, ~band)
            near_score, near_class = metric(fixed_near, target)
            far_score, far_class = metric(fixed_far, target)
            record[f"within_{radius}px"] = {
                "gt_boundary_band_pixels": int(band.sum()),
                "misclassified_pixels_in_band": corrected_near,
                "fraction_of_all_pixel_errors_in_band": (
                    corrected_near / record["misclassified_pixels"]
                    if record["misclassified_pixels"] else 0.0
                ),
                "oracle_corrected_image_miou_empty0": near_score,
                "oracle_corrected_class_iou": dict(zip(
                    spatial.audit.model350.CLASS_ORDER, near_class.tolist())),
            }
            record[f"outside_{radius}px"] = {
                "misclassified_pixels_outside_band": corrected_far,
                "oracle_corrected_image_miou_empty0": far_score,
                "oracle_corrected_class_iou": dict(zip(
                    spatial.audit.model350.CLASS_ORDER, far_class.tolist())),
            }
        per_image.append(record)
        print(f"audited {index + 1}/{len(rows)}: {name}", flush=True)

    image_scores = np.asarray([r["baseline_image_miou_empty0"] for r in per_image])
    archived = np.asarray(model360["raw_best_image_scores"], dtype=np.float64)
    if names != model360["image_names"] or not np.allclose(
            image_scores, archived, atol=1e-10, rtol=0):
        raise AssertionError("Model360 per-image prediction parity failed")
    summary = {
        "audit": "GT-oracle boundary correction diagnostic; not deployable",
        "metric": "20-image mean of four class IoUs; empty union=0",
        "model": 360,
        "class_order": list(spatial.audit.model350.CLASS_ORDER),
        "boundary_definition": "class-transition pixels dilated by radius with 8-connectivity",
        "baseline_image_mean_miou_empty0": float(image_scores.mean()),
        "pixel_error_fraction_within_boundary_band": {},
        "oracle_image_mean_miou_if_correct_all_errors_within_band": {},
        "oracle_image_mean_miou_if_correct_only_errors_outside_band": {},
        "per_image": per_image,
    }
    for radius in args.radii:
        field = f"within_{radius}px"
        summary["pixel_error_fraction_within_boundary_band"][str(radius)] = float(
            sum(r[field]["misclassified_pixels_in_band"] for r in per_image)
            / sum(r["misclassified_pixels"] for r in per_image)
        )
        summary["oracle_image_mean_miou_if_correct_all_errors_within_band"][str(radius)] = float(
            np.mean([r[field]["oracle_corrected_image_miou_empty0"] for r in per_image])
        )
        summary["oracle_image_mean_miou_if_correct_only_errors_outside_band"][str(radius)] = float(
            np.mean([r[f"outside_{radius}px"]["oracle_corrected_image_miou_empty0"]
                     for r in per_image])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "baseline_image_mean_miou_empty0", "pixel_error_fraction_within_boundary_band",
        "oracle_image_mean_miou_if_correct_all_errors_within_band",
        "oracle_image_mean_miou_if_correct_only_errors_outside_band")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
