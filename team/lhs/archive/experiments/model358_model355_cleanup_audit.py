#!/usr/bin/env python3
"""Audit Primary-Si component cleanup thresholds on fixed Model355 route."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import model355_model349_spatial_audit as spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    outputs = args.root / "outputs"
    rows, names, base_route, gamma329, _ = spatial.audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    _, probs347 = spatial.audit.model350.model341.route.load_archive(
        outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        team_order=True)
    if set(probs347) != set(names):
        raise ValueError("Model347 validation image-key mismatch")
    for name, row in zip(names, rows):
        row["sources"]["347"] = probs347[name]
    model346 = json.loads((outputs / "model346_model336_into_341_route/results.json")
                           .read_text(encoding="utf-8"))
    model349 = json.loads((outputs / "model349_model347_into_346_route/results.json")
                           .read_text(encoding="utf-8"))
    model355 = json.loads((outputs / "model355_model349_spatial_audit/results.json")
                           .read_text(encoding="utf-8"))
    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    g, e = model355["raw_best_spatial_blend"]
    thresholds = (128, 0, 32, 64, 256, 512)
    scores = np.empty((len(thresholds), len(rows)), dtype=np.float64)
    classes = np.empty((len(thresholds), len(rows), 4), dtype=np.float64)
    for image_index, row in enumerate(rows):
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        fixed_route = dict(base_route, gaussian_blend=g, eut_edge_blend=e)
        baseline = spatial.prediction(parts, fixed_route, g, e)
        expected = spatial.audit.model350.model346._candidate_prediction(
            row, (0.0, 0.0, 0.0, 0.0), fixed_route, gamma329, parts[4],
            gamma336, gamma347)
        if not np.array_equal(baseline, expected):
            raise AssertionError(f"Model355 pixel parity failed for {names[image_index]}")
        target = row["item"]["target"]
        for setting_index, threshold in enumerate(thresholds):
            route = dict(fixed_route, cleanup_threshold=threshold)
            pred = baseline if setting_index == 0 else spatial.prediction(parts, route, g, e)
            classes[setting_index, image_index] = spatial.audit.class_iou(pred, target)
            scores[setting_index, image_index] = classes[setting_index, image_index].mean()
        print(f"scored {image_index + 1}/{len(rows)}: {names[image_index]}", flush=True)
    reference = float(scores[0].mean())
    archived = float(model355["raw_best_valid_image_mean_miou_empty0"])
    if not np.isclose(reference, archived, rtol=0, atol=1e-10):
        raise AssertionError(f"Model355 score parity failed: {reference} vs {archived}")

    groups = {}
    for index, name in enumerate(names):
        match = re.match(r"(A\d+)_", name)
        if not match:
            raise ValueError(f"cannot parse specimen group: {name}")
        groups.setdefault(match.group(1), []).append(index)
    selected = np.empty(len(names), dtype=np.int64)
    choices = []
    for group, indices in groups.items():
        other = [i for i in range(len(names)) if i not in indices]
        best = int(np.argmax(scores[:, other].mean(axis=1)))
        selected[indices] = best
        choices.append({"heldout_group": group, "cleanup_threshold": thresholds[best]})
    held_scores = scores[selected, np.arange(len(names))]
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": 358,
        "metric": "20-image mean of four class IoUs; empty union=0",
        "fixed_model355_spatial_blend": [g, e],
        "baseline_model355": reference,
        "raw_best_cleanup_threshold": thresholds[best],
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": classes[best].mean(axis=0).tolist(),
        "group_cv_heldout_miou_empty0": float(held_scores.mean()),
        "group_cv_delta_pp": float((held_scores - scores[0]).mean() * 100.0),
        "group_cv_choices": choices,
        "settings": [{"cleanup_threshold": threshold, "score": float(score)}
                     for threshold, score in zip(thresholds, scores.mean(axis=1))],
        "image_names": names,
        "baseline_image_scores": scores[0].tolist(),
        "raw_best_image_scores": scores[best].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model355", "raw_best_cleanup_threshold",
        "raw_best_valid_image_mean_miou_empty0", "group_cv_heldout_miou_empty0",
        "group_cv_delta_pp")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
