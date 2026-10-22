#!/usr/bin/env python3
"""Cache-aware class-specific branch fusion search.

The old narrow search recalculates the guide-dependent bilateral weights for
every candidate.  Those weights depend only on the image z-score guide, so
this implementation builds them once per validation image and reuses them.
It first selects Primary candidates, then searches rare classes around the
strongest candidates, and finally does a small calibration pass.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from model181_raw_zcdf_aspp import score
from model183_metric_calibration import prior_terms
from model187_schedule_branch_blend import undo_model181_primary_blend


def replace_classes(base: np.ndarray, source: np.ndarray | None,
                    classes: tuple[int, ...], gamma: float) -> np.ndarray:
    if source is None or abs(gamma) < 1e-12 or not classes:
        return base
    output = base.copy()
    for class_id in classes:
        output[class_id] = (1.0 - gamma) * base[class_id] + gamma * source[class_id]
    return output / np.maximum(output.sum(axis=0, keepdims=True), 1e-7)


def make_mixed(row: dict, setting: dict) -> np.ndarray:
    mixed = row["base"]
    mixed = replace_classes(mixed, row["sources"][setting["primary_source"]],
                            (1,), setting["primary_gamma"])
    if (setting["al3ni_source"] == setting["eutectic_source"]
            and setting["al3ni_gamma"] == setting["eutectic_gamma"]):
        return replace_classes(mixed, row["sources"][setting["al3ni_source"]],
                               (0, 3), setting["al3ni_gamma"])
    mixed = replace_classes(mixed, row["sources"][setting["al3ni_source"]],
                            (0,), setting["al3ni_gamma"])
    return replace_classes(mixed, row["sources"][setting["eutectic_source"]],
                           (3,), setting["eutectic_gamma"])


def bilateral_weights(guide: np.ndarray, radius: int = 2,
                      sigma_spatial: float = 1.0,
                      sigma_range: float = 0.60) -> list[np.ndarray]:
    pad = radius
    guide_pad = np.pad(guide, pad, mode="reflect")
    center = guide_pad[pad:pad + guide.shape[0], pad:pad + guide.shape[1]]
    spatial_denominator = 2.0 * sigma_spatial * sigma_spatial
    range_denominator = 2.0 * sigma_range * sigma_range
    weights = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_distance = float(dx * dx + dy * dy)
            spatial_weight = np.exp(-spatial_distance / spatial_denominator)
            neighbour_guide = guide_pad[
                pad + dy:pad + dy + guide.shape[0],
                pad + dx:pad + dx + guide.shape[1],
            ]
            weights.append((spatial_weight * np.exp(
                -((neighbour_guide - center) ** 2) / range_denominator
            )).astype(np.float32))
    return weights


def apply_bilateral(field: np.ndarray, guide_weights: list[np.ndarray],
                    radius: int = 2) -> np.ndarray:
    pad = radius
    field_pad = np.pad(field, pad, mode="reflect")
    numerator = np.zeros_like(field, dtype=np.float32)
    denominator = np.zeros_like(field, dtype=np.float32)
    index = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbour = field_pad[
                pad + dy:pad + dy + field.shape[0],
                pad + dx:pad + dx + field.shape[1],
            ]
            weight = guide_weights[index]
            numerator += weight * neighbour
            denominator += weight
            index += 1
    return numerator / np.maximum(denominator, 1e-7)


def predict(item: dict, mixed: np.ndarray, setting: dict,
            weights: list[np.ndarray], exact: bool) -> np.ndarray:
    base_prior, delta = prior_terms(item, mixed)
    prior = setting["prior_scale"] * base_prior
    prior[0] += (setting["prior_scale"] * setting["beta_scale"]
                 * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + setting["al_delta"]
    logits[3] += item["params"]["eut_bias"] + setting["eut_delta"]
    if exact:
        refined = logits.copy()
        refined[1] = (setting["gaussian_blend"] * gaussian_filter(
            logits[1], sigma=0.50, mode="reflect")
            + (1.0 - setting["gaussian_blend"]) * apply_bilateral(logits[1], weights))
        refined[3] = ((1.0 - setting["eut_edge_blend"]) * logits[3]
                      + setting["eut_edge_blend"] * apply_bilateral(logits[3], weights))
        prediction = cleanup_primary(
            refined.argmax(axis=0).astype(np.uint8), refined,
            setting["cleanup_threshold"])
        if float((prediction == 1).mean()) <= 0.01:
            mask = prediction == 1
            alternative = refined.copy()
            alternative[1] = -np.inf
            prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
    else:
        # The first pass only ranks branch/calibration candidates.  It avoids
        # the expensive spatial filters; all finalists are re-evaluated below
        # with the exact model-197 route.
        prediction = logits.argmax(axis=0).astype(np.uint8)
    return prediction


def evaluate(rows: list[dict], settings: list[dict], exact: bool) -> list[dict]:
    n = len(settings)
    matrices = np.zeros((n, 4, 4), dtype=np.int64)
    image_scores = np.zeros((n, len(rows)), dtype=np.float64)
    image_scores_exclude = np.zeros_like(image_scores)
    for row in rows:
        # The cheap ranking pass never uses spatial refinement.  Building the
        # 25-neighbour bilateral cache there made the old search needlessly
        # expensive and obscured whether branch fusion itself was useful.
        weights = bilateral_weights(row["item"]["z"]) if exact else None
        target = row["item"]["target"]
        for index, candidate in enumerate(settings):
            mixed = make_mixed(row, candidate)
            prediction = predict(row["item"], mixed, candidate, weights, exact)
            bins = target.reshape(-1) * 4 + prediction.reshape(-1)
            matrix = np.bincount(bins, minlength=16).reshape(4, 4)
            matrices[index] += matrix
            intersection = np.diag(matrix).astype(np.float64)
            union = matrix.sum(0) + matrix.sum(1) - intersection
            iou = intersection / np.maximum(union, 1.0)
            image_scores[index, row["index"]] = iou.mean()
            image_scores_exclude[index, row["index"]] = iou[union > 0].mean()
    records = []
    for index, candidate in enumerate(settings):
        total = matrices[index]
        intersection = np.diag(total).astype(np.float64)
        union = total.sum(0) + total.sum(1) - intersection
        image_class_iou = intersection / np.maximum(union, 1.0)
        records.append({
            **candidate,
            "image_miou_compat": float(image_class_iou.mean()),
            "image_class_iou": {name: float(value) for name, value in zip(
                ("Al3Ni", "Primary Si", "Al", "Eutectic Si"), image_class_iou)},
            "image_mean_miou": float(image_scores[index].mean()),
            "image_mean_miou_exclude": float(image_scores_exclude[index].mean()),
            "image_scores": image_scores[index].tolist(),
            "image_scores_exclude": image_scores_exclude[index].tolist(),
            "confusion": total.tolist(),
        })
    return records


def default_setting(**kwargs: object) -> dict:
    result = {
        "primary_source": "195", "primary_gamma": 0.75,
        "al3ni_source": "195", "al3ni_gamma": 0.075,
        "eutectic_source": "195", "eutectic_gamma": 0.075,
        "prior_scale": 1.25, "beta_scale": 0.80,
        "al_delta": 0.0, "eut_delta": 0.0,
        "gaussian_blend": 0.50, "eut_edge_blend": 0.25,
        "cleanup_threshold": 128,
    }
    result.update(kwargs)
    return result


def top(records: list[dict], key: str, count: int) -> list[dict]:
    return sorted(records, key=lambda row: row[key], reverse=True)[:count]


def main(args: argparse.Namespace) -> None:
    entries = pickle.load(Path(args.entries).open("rb"))
    source_paths = {
        "186": args.model186,
        "192": args.model192,
        "195": args.model195,
        "199": args.model199,
    }
    source_values: dict[str, list[np.ndarray]] = {}
    for name, path in source_paths.items():
        archive = np.load(path)
        source_values[name] = [archive[key].astype(np.float32) for key in archive.files]
    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "index": index,
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{name: values[index] for name, values in source_values.items()},
            },
        })

    records: list[dict] = []
    source_names = ("base", "181", "186", "192", "195", "199")
    primary_settings = [default_setting(primary_source=source, primary_gamma=gamma)
                        for source in source_names
                        for gamma in (0.0, 0.50, 0.65, 0.75, 0.90, 1.0)]
    stage = evaluate(rows, primary_settings, exact=False)
    records.extend(stage)
    parents = top(stage, "image_mean_miou", 4)

    rare_settings = []
    rare_sources = ("base", "186", "192", "195", "199")
    for parent in parents:
        for class_name in ("al3ni", "eutectic"):
            for source in rare_sources:
                for gamma in (0.0, 0.075, 0.15, 0.25):
                    candidate = dict(parent)
                    candidate["stage"] = f"{class_name}_coordinate"
                    candidate[f"{class_name}_source"] = source
                    candidate[f"{class_name}_gamma"] = gamma
                    rare_settings.append(candidate)
    stage = evaluate(rows, rare_settings, exact=False)
    records.extend(stage)
    parents = top(stage, "image_mean_miou", 4)

    joint_settings = []
    for parent in parents:
        # Recombine the best rare strengths with all source pairings.  This is
        # cheap after the guide cache and catches complementary rare branches.
        for al_source in rare_sources:
            for eut_source in rare_sources:
                candidate = dict(parent)
                candidate["stage"] = "joint_rare_sources"
                candidate["al3ni_source"] = al_source
                candidate["eutectic_source"] = eut_source
                joint_settings.append(candidate)
    stage = evaluate(rows, joint_settings, exact=False)
    records.extend(stage)
    parents = top(stage, "image_mean_miou", 4)

    calibration_settings = []
    for parent in parents:
        for field, values in (
            ("prior_scale", (1.15, 1.25, 1.35)),
            ("beta_scale", (0.60, 0.80, 1.00)),
            ("al_delta", (0.0, 0.10)),
            ("eut_delta", (-0.15, 0.0)),
            ("gaussian_blend", (0.25, 0.50)),
            ("eut_edge_blend", (0.15, 0.25)),
            ("cleanup_threshold", (128, 256)),
        ):
            for value in values:
                candidate = dict(parent)
                candidate["stage"] = f"calibration_{field}"
                candidate[field] = value
                calibration_settings.append(candidate)
    stage = evaluate(rows, calibration_settings, exact=False)
    records.extend(stage)

    # Only exact-evaluate the strongest cheap-pass candidates.  Include the
    # documented model-197 setting explicitly as a guard against a fast-pass
    # ranking artifact.
    exact_settings = [default_setting()]
    # `records` contains the cheap-pass metrics; the raw setting lists do not.
    # Passing the raw settings here caused the previous long run to fail with
    # a KeyError when it tried to rank the exact finalists.
    exact_candidates = [row for row in records if "image_mean_miou" in row]
    exact_settings.extend(top(exact_candidates, "image_mean_miou", 24))
    exact_stage = evaluate(rows, exact_settings, exact=True)
    records.extend(exact_stage)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "record_count": len(records),
        "best_image": max(records, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(records, key=lambda row: row["image_mean_miou_exclude"]),
        "best_image_compat": max(records, key=lambda row: row["image_miou_compat"]),
        "records": records,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "record_count", "best_image", "best_exclude", "best_image_compat")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model186", required=True)
    parser.add_argument("--model192", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--model199", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
