#!/usr/bin/env python3
"""Search Model349's two spatial-refinement blends without retraining.

The existing Model349 class route, prior, cleanup, and checkpoints are fixed.
Only the Primary-Si Gaussian/bilateral mix and Eutectic-Si edge-bilateral mix
change. The original setting must match Model349 pixel-for-pixel.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import model354_model349_bias_audit as audit


def spatial_parts(row, base_route, gamma329, gamma336, gamma347):
    route = audit.model350.model341.route
    mixed = route.make_mixed(row, base_route)
    for key, gammas in (("329", gamma329), ("336", gamma336), ("347", gamma347)):
        for class_id, gamma in enumerate(gammas):
            if gamma:
                mixed = route.replace_classes(mixed, row["sources"][key],
                                              (class_id,), gamma)
    item = row["item"]
    base_prior, delta = audit.prior_terms(item, mixed)
    prior = base_route["prior_scale"] * base_prior
    prior[0] += (base_route["prior_scale"] * base_route["beta_scale"]
                 * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + base_route["al_delta"]
    logits[3] += item["params"]["eut_bias"] + base_route["eut_delta"]
    weights = route.bilateral_weights(item["z"])
    primary_gaussian = gaussian_filter(logits[1], sigma=0.50, mode="reflect")
    primary_bilateral = audit.apply_bilateral(logits[1], weights)
    eut_bilateral = audit.apply_bilateral(logits[3], weights)
    return logits, primary_gaussian, primary_bilateral, eut_bilateral, weights


def prediction(parts, base_route, gaussian_blend, eut_edge_blend):
    logits, primary_gaussian, primary_bilateral, eut_bilateral, _ = parts
    refined = logits.copy()
    refined[1] = (gaussian_blend * primary_gaussian
                  + (1.0 - gaussian_blend) * primary_bilateral)
    refined[3] = ((1.0 - eut_edge_blend) * logits[3]
                  + eut_edge_blend * eut_bilateral)
    return audit.predict_from_refined(refined, base_route, 0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", type=int, default=355)
    parser.add_argument("--gaussian-grid", type=float, nargs="+",
                        default=(0.0, 0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--eut-grid", type=float, nargs="+",
                        default=(0.0, 0.125, 0.25, 0.375, 0.50))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    outputs = args.root / "outputs"
    rows, names, base_route, gamma329, _ = audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    _, probs347 = audit.model350.model341.route.load_archive(
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
    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    baseline_setting = (base_route["gaussian_blend"], base_route["eut_edge_blend"])
    gaussian_grid = args.gaussian_grid
    eut_grid = args.eut_grid
    settings = [baseline_setting] + [(g, e) for g in gaussian_grid for e in eut_grid
                                     if (g, e) != baseline_setting]
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    classes = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    for image_index, row in enumerate(rows):
        parts = spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        expected = audit.model350.model346._candidate_prediction(
            row, (0.0, 0.0, 0.0, 0.0), base_route, gamma329, parts[4],
            gamma336, gamma347)
        baseline = prediction(parts, base_route, *baseline_setting)
        if not np.array_equal(baseline, expected):
            raise AssertionError(f"Model349 pixel parity failed for {names[image_index]}")
        target = row["item"]["target"]
        for setting_index, setting in enumerate(settings):
            pred = baseline if setting_index == 0 else prediction(parts, base_route, *setting)
            classes[setting_index, image_index] = audit.class_iou(pred, target)
            scores[setting_index, image_index] = classes[setting_index, image_index].mean()
        print(f"scored {image_index + 1}/{len(rows)}: {names[image_index]}", flush=True)
    reference = float(scores[0].mean())
    archived = float(model349["raw_best_valid_image_mean_miou_empty0"])
    if not np.isclose(reference, archived, rtol=0, atol=1e-10):
        raise AssertionError(f"Model349 score parity failed: {reference} vs {archived}")

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
        choices.append({"heldout_group": group, "spatial_blend": settings[best]})
    held_scores = scores[selected, np.arange(len(names))]
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": args.model_id,
        "metric": "20-image mean of four class IoUs; empty union=0",
        "baseline_model349": reference,
        "baseline_spatial_blend": baseline_setting,
        "raw_best_spatial_blend": settings[best],
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": classes[best].mean(axis=0).tolist(),
        "group_cv_heldout_miou_empty0": float(held_scores.mean()),
        "group_cv_delta_pp": float((held_scores - scores[0]).mean() * 100.0),
        "group_cv_choices": choices,
        "settings": [{"spatial_blend": setting, "score": float(score)}
                     for setting, score in zip(settings, scores.mean(axis=1))],
        "image_names": names,
        "baseline_image_scores": scores[0].tolist(),
        "raw_best_image_scores": scores[best].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model349", "raw_best_spatial_blend",
        "raw_best_valid_image_mean_miou_empty0", "group_cv_heldout_miou_empty0",
        "group_cv_delta_pp")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
