#!/usr/bin/env python3
"""Decompose saved Model380 route scores; performs no new route search."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


GROUP_RE = re.compile(r"^(A\d+)_")
CLASS_NAMES = ("Al3Ni", "Primary_Si", "Al", "Eutectic_Si")


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        scores = np.asarray(archive["image_scores"], dtype=np.float64)
        classes = np.asarray(archive["class_scores"], dtype=np.float64)
        names = [str(value) for value in archive["image_names"].tolist()]
        settings = np.asarray(archive["gamma_settings"], dtype=np.float64)
    if scores.ndim != 2 or classes.shape != (*scores.shape, 4):
        raise ValueError(f"bad matrix shapes in {path}: {scores.shape}, {classes.shape}")
    if scores.shape[0] != len(names) or scores.shape[1] != len(settings):
        raise ValueError(f"matrix/name/setting count mismatch in {path}")
    if not np.isfinite(scores).all() or not np.isfinite(classes).all():
        raise ValueError(f"non-finite score in {path}")
    return scores, classes, names, settings


def setting_index(settings: np.ndarray, value: list[float]) -> int:
    wanted = np.asarray(value, dtype=np.float64)
    matches = np.flatnonzero(np.all(np.isclose(settings, wanted, atol=1e-7, rtol=0), axis=1))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one setting {wanted.tolist()}, found {len(matches)}")
    return int(matches[0])


def run(audit_path: Path, global_path: Path, alternate_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    global_scores, global_classes, names, global_settings = load_matrix(global_path)
    alt_scores, alt_classes, alt_names, alt_settings = load_matrix(alternate_path)
    if names != alt_names or len(names) != 20:
        raise ValueError("global and alternate matrices must have identical ordered 20-image IDs")

    full336 = np.asarray(audit["gamma336_full_valid"], dtype=np.float64)
    choices336 = audit["gamma336_choices_by_heldout_group"]
    choices347 = audit["gamma347_choices_by_heldout_group"]
    mode347_tuple, mode_count = Counter(tuple(v) for v in choices347.values()).most_common(1)[0]
    if sum(tuple(v) == mode347_tuple for v in choices347.values()) != mode_count:
        raise ValueError("ambiguous modal full-valid gamma347 selection")
    full347 = list(mode347_tuple)
    nonmodal336 = {tuple(v) for v in choices336.values() if not np.allclose(v, full336, atol=1e-7, rtol=0)}
    if len(nonmodal336) > 1:
        raise ValueError("alternate matrix does not identify a unique non-modal gamma336")
    alt336 = next(iter(nonmodal336), None)

    zero347 = [0.0, 0.0, 0.0, 0.0]
    rows: list[dict[str, Any]] = []
    for image_index, image_id in enumerate(names):
        match = GROUP_RE.match(image_id)
        if not match:
            raise ValueError(f"cannot parse specimen group in {image_id}")
        group = match.group(1)
        gamma336 = choices336[group]
        gamma347 = choices347[group]
        is_alt = not np.allclose(gamma336, full336, atol=1e-7, rtol=0)
        if is_alt and (alt336 is None or not np.allclose(gamma336, alt336, atol=1e-7, rtol=0)):
            raise ValueError(f"no matching alternate matrix for {group}: {gamma336}")
        cf_scores, cf_classes, cf_settings = (
            (alt_scores, alt_classes, alt_settings) if is_alt
            else (global_scores, global_classes, global_settings)
        )

        cf_reference_index = setting_index(cf_settings, zero347)
        cf_candidate_index = setting_index(cf_settings, gamma347)
        full_reference_index = setting_index(global_settings, zero347)
        full_candidate_index = setting_index(global_settings, full347)

        cf_reference = float(cf_scores[image_index, cf_reference_index])
        cf_candidate = float(cf_scores[image_index, cf_candidate_index])
        full_reference = float(global_scores[image_index, full_reference_index])
        full_candidate = float(global_scores[image_index, full_candidate_index])
        class_cf_delta = (cf_classes[image_index, cf_candidate_index]
                          - cf_classes[image_index, cf_reference_index])
        class_selection_delta = (cf_classes[image_index, cf_candidate_index]
                                 - global_classes[image_index, full_candidate_index])
        route_changed = (is_alt or not np.allclose(gamma347, full347, atol=1e-7, rtol=0))

        rows.append({
            "image": image_id,
            "group": group,
            "gamma336_crossfit": list(gamma336),
            "gamma347_crossfit": list(gamma347),
            "route_setting_differs_from_full_valid": bool(route_changed),
            "crossfit_candidate_minus_crossfit_reference_miou_pp":
                100.0 * (cf_candidate - cf_reference),
            "crossfit_candidate_minus_full_valid_candidate_miou_pp":
                100.0 * (cf_candidate - full_candidate),
            "full_valid_candidate_minus_full_valid_reference_miou_pp":
                100.0 * (full_candidate - full_reference),
            "crossfit_candidate_minus_crossfit_reference_class_iou_pp": {
                name: float(100.0 * class_cf_delta[i]) for i, name in enumerate(CLASS_NAMES)
            },
            "crossfit_candidate_minus_full_valid_candidate_class_iou_pp": {
                name: float(100.0 * class_selection_delta[i]) for i, name in enumerate(CLASS_NAMES)
            },
        })

    crossfit_delta = float(np.mean([
        row["crossfit_candidate_minus_crossfit_reference_miou_pp"] for row in rows
    ]))
    full_delta = float(np.mean([
        row["full_valid_candidate_minus_full_valid_reference_miou_pp"] for row in rows
    ]))
    selection_delta = float(np.mean([
        row["crossfit_candidate_minus_full_valid_candidate_miou_pp"] for row in rows
    ]))
    expected_cf = float(audit["paired_outer_group_comparison_349_vs_346"][
        "paired_image_weighted_delta_pp"
    ])
    if abs(crossfit_delta - expected_cf) > 1e-6:
        raise ValueError(f"cross-fit delta does not reproduce Model380: {crossfit_delta} vs {expected_cf}")
    if abs(float(np.mean([global_scores[i, setting_index(global_settings, full347)]
                          for i in range(len(names))])) * 100.0
           - float(audit["raw_full_valid_model349_pct"])) > 1e-5:
        raise ValueError("modal full-valid route does not reproduce recorded Model349 score")
    if abs(float(np.mean([global_scores[i, full_reference_index]
                          for i in range(len(names))])) * 100.0
           - float(audit["raw_full_valid_model346_pct"])) > 1e-5:
        raise ValueError("full-valid gamma347=0 route does not reproduce recorded Model346 score")

    return {
        "model": 380,
        "purpose": "read-only decomposition of saved route matrices; no new search or training",
        "scope": "20 official validation images, 19 specimen groups; conditional on the historical Model341 anchor",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "full_valid_gamma336": full336.tolist(),
        "full_valid_gamma347_modal": full347,
        "crossfit_incremental_delta_pp": crossfit_delta,
        "full_valid_incremental_delta_pp": full_delta,
        "full_valid_minus_crossfit_incremental_delta_pp": full_delta - crossfit_delta,
        "crossfit_candidate_minus_full_valid_candidate_pp": selection_delta,
        "groups_with_parameter_choice_different_from_full_valid": sorted({
            row["group"] for row in rows if row["route_setting_differs_from_full_valid"]
        }),
        "images_with_parameter_choice_different_from_full_valid": sum(
            row["route_setting_differs_from_full_valid"] for row in rows
        ),
        "interpretation": [
            "The cross-fit-minus-reference delta is the conditional held-out estimate for adding gamma347 on each fold's gamma336 route.",
            "The full-valid-minus-crossfit gap is descriptive selection sensitivity, not an independent-test estimate; Model341 anchor selection remains uncorrected.",
            "Per-image route-choice differences occur only where the cross-fit gamma settings differ from the full-valid setting; do not infer generalizable routing bias from this 20-image pool.",
        ],
        "per_image": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model380-json", type=Path, required=True)
    parser.add_argument("--global-matrix", type=Path, required=True)
    parser.add_argument("--alternate-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(args.model380_json, args.global_matrix, args.alternate_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
