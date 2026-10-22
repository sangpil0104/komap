#!/usr/bin/env python3
"""Gate scale-3 HRNet probabilities into the fixed Model360 D4 mixture.

This is a same-validation diagnostic. The zero-strength reference must
reproduce Model360 pixel masks and its official image-wise empty=0 score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import model360_entropy_gated_d4_into_355 as model360


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-probabilities", type=Path, default=None,
                        help="Optional alternate Model361 TTA probability archive.")
    parser.add_argument("--model-id", type=int, default=364)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    spatial = model360.spatial
    outputs = args.root / "outputs"
    rows, names, route, gamma329, _ = spatial.audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    load = spatial.audit.model350.model341.route.load_archive
    archives = {}
    for key, path in (
        ("347", outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"),
        ("351", outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"),
        ("361", args.expert_probabilities or
         outputs / "model361_hrnet_scale3_finetune/valid_eval/valid_probabilities.npz"),
    ):
        _, archives[key] = load(path, team_order=True)
        if set(archives[key]) != set(names):
            raise ValueError(f"Model{key} image keys mismatch")
    result346 = json.loads((outputs / "model346_model336_into_341_route/results.json").read_text())
    result349 = json.loads((outputs / "model349_model347_into_346_route/results.json").read_text())
    result355 = json.loads((outputs / "model355_model349_spatial_audit/results.json").read_text())
    result360 = json.loads((outputs / "model360_entropy_gated_d4_into_355/results.json").read_text())
    gamma336 = tuple(result346["raw_best_gamma336"])
    gamma347 = tuple(result349["raw_best_gamma347"])
    g, e = result355["raw_best_spatial_blend"]
    d4_margin, d4_strength = result360["raw_best_entropy_gate"]
    fixed_route = dict(route, gaussian_blend=g, eut_edge_blend=e)
    settings = [(0.0, 0.0)] + [
        (margin, strength) for margin in (0.0, 0.05, 0.10)
        for strength in (0.25, 0.50, 1.0)
    ]
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    classes = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    coverage = np.empty((3, len(rows)), dtype=np.float64)
    for image_index, (name, row) in enumerate(zip(names, rows)):
        p336 = row["sources"]["336"]
        p351 = archives["351"][name]
        p361 = archives["361"][name]
        if p336.shape != p351.shape or p336.shape != p361.shape:
            raise ValueError(f"probability shapes mismatch: {name}")
        row["sources"]["347"] = archives["347"][name]
        d4_gate = (model360.entropy(p351) + d4_margin
                   < model360.entropy(p336)).astype(np.float32)
        d4_base = p336 + d4_strength * d4_gate[None] * (p351 - p336)
        gates = [(model360.entropy(p361) + margin < model360.entropy(d4_base))
                 for margin in (0.0, 0.05, 0.10)]
        coverage[:, image_index] = [float(gate.mean()) for gate in gates]
        target = row["item"]["target"]
        for setting_index, (margin, strength) in enumerate(settings):
            if setting_index == 0:
                row["sources"]["336"] = d4_base
            else:
                gate = gates[(0.0, 0.05, 0.10).index(margin)]
                row["sources"]["336"] = d4_base + strength * gate[None] * (p361 - d4_base)
            parts = spatial.spatial_parts(row, route, gamma329, gamma336, gamma347)
            pred = spatial.prediction(parts, fixed_route, g, e)
            classes[setting_index, image_index] = spatial.audit.class_iou(pred, target)
            scores[setting_index, image_index] = classes[setting_index, image_index].mean()
        row["sources"]["336"] = p336
        print(f"scored {image_index + 1}/{len(rows)}: {name}", flush=True)
    reference = float(scores[0].mean())
    archived = float(result360["raw_best_valid_image_mean_miou_empty0"])
    if names != result360["image_names"]:
        raise AssertionError("Model360 image ordering changed")
    if not np.allclose(scores[0], result360["raw_best_image_scores"], rtol=0, atol=1e-10):
        raise AssertionError("Model360 per-image score parity failed")
    if not np.isclose(reference, archived, rtol=0, atol=1e-10):
        raise AssertionError(f"Model360 score parity failed: {reference} vs {archived}")
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
        choices.append({"heldout_group": group, "selected_gate": settings[best]})
    held_scores = scores[selected, np.arange(len(names))]
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": args.model_id,
        "metric": "20-image mean of four class IoUs; empty union=0",
        "baseline_model360": reference,
        "fixed_model360_d4_gate": [d4_margin, d4_strength],
        "expert_probabilities": str(args.expert_probabilities) if args.expert_probabilities else
                                str(outputs / "model361_hrnet_scale3_finetune/valid_eval/valid_probabilities.npz"),
        "raw_best_scale3_gate": settings[best],
        "raw_best_expert_gate": settings[best],
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": classes[best].mean(axis=0).tolist(),
        "group_cv_heldout_miou_empty0": float(held_scores.mean()),
        "group_cv_delta_pp": float((held_scores - scores[0]).mean() * 100),
        "group_cv_choices": choices,
        "gate_coverage_mean": coverage.mean(axis=1).tolist(),
        "settings": [{"gate": setting, "score": float(score)}
                     for setting, score in zip(settings, scores.mean(axis=1))],
        "image_names": names,
        "baseline_image_scores": scores[0].tolist(),
        "raw_best_image_scores": scores[best].tolist(),
        "warning": "Same 20 valid images used for model and gate selection; not an independent test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model360", "raw_best_scale3_gate",
        "raw_best_valid_image_mean_miou_empty0", "group_cv_heldout_miou_empty0",
        "group_cv_delta_pp")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
