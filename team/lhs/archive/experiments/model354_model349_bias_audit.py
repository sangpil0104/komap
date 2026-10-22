#!/usr/bin/env python3
"""Audit small fixed-route logit-bias shifts on Model349, with specimen CV.

The exact Model349 mixed probabilities and spatial refinement are cached once
per validation image. Constant class biases are then added to the refined
logits, so the zero-bias candidate must reproduce the saved Model349 score.
No new model is trained and no test labels are used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model350_model346_fine_gamma as model350  # noqa: E402
from aspp768_addition import cleanup_primary  # noqa: E402
from model183_metric_calibration import prior_terms  # noqa: E402
from model198_fast_branch_search import apply_bilateral  # noqa: E402


def build_refined_logits(row, base_route, gamma329, gamma336, gamma347):
    route = model350.model341.route
    mixed = route.make_mixed(row, base_route)
    for key, gammas in (("329", gamma329), ("336", gamma336), ("347", gamma347)):
        for class_id, gamma in enumerate(gammas):
            if gamma:
                mixed = route.replace_classes(mixed, row["sources"][key],
                                              (class_id,), gamma)
    item = row["item"]
    base_prior, delta = prior_terms(item, mixed)
    prior = base_route["prior_scale"] * base_prior
    prior[0] += (base_route["prior_scale"] * base_route["beta_scale"]
                 * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + base_route["al_delta"]
    logits[3] += item["params"]["eut_bias"] + base_route["eut_delta"]
    weights = route.bilateral_weights(item["z"])
    refined = logits.copy()
    refined[1] = (base_route["gaussian_blend"] * gaussian_filter(
        logits[1], sigma=0.50, mode="reflect")
        + (1.0 - base_route["gaussian_blend"])
        * apply_bilateral(logits[1], weights))
    refined[3] = ((1.0 - base_route["eut_edge_blend"]) * logits[3]
                  + base_route["eut_edge_blend"]
                  * apply_bilateral(logits[3], weights))
    return refined, weights


def predict_from_refined(refined, base_route, al_bias, eut_bias):
    logits = refined.copy()
    logits[2] += al_bias
    logits[3] += eut_bias
    prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits,
                                 base_route["cleanup_threshold"])
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = logits.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def class_iou(prediction, target):
    matrix = np.bincount(target.astype(np.int64).ravel() * 4
                         + prediction.astype(np.int64).ravel(),
                         minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", type=int, default=354)
    parser.add_argument("--gaussian-blend", type=float, default=None)
    parser.add_argument("--eut-edge-blend", type=float, default=None)
    parser.add_argument("--reference-json", type=Path, default=None,
                        help="previous fixed-spatial-route result for zero-bias parity")
    parser.add_argument("--bias-grid", type=float, nargs="+",
                        default=(-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    outputs = args.root / "outputs"
    expert_path = (outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    rows, names, base_route, gamma329, _ = model350.load_rows(args.root, expert_path)
    _, probs347 = model350.model341.route.load_archive(
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
    base_route = dict(base_route)
    if args.gaussian_blend is not None:
        base_route["gaussian_blend"] = args.gaussian_blend
    if args.eut_edge_blend is not None:
        base_route["eut_edge_blend"] = args.eut_edge_blend
    shifts = args.bias_grid
    settings = [(0.0, 0.0)] + [(a, e) for a in shifts for e in shifts
                               if (a, e) != (0.0, 0.0)]
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    classes = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    for image_index, row in enumerate(rows):
        refined, weights = build_refined_logits(row, base_route, gamma329,
                                                gamma336, gamma347)
        expected = model350.model346._candidate_prediction(
            row, (0.0, 0.0, 0.0, 0.0), base_route, gamma329, weights,
            gamma336, gamma347)
        actual = predict_from_refined(refined, base_route, 0.0, 0.0)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"zero-bias pixel parity failed for {names[image_index]}")
        target = row["item"]["target"]
        for setting_index, (a, e) in enumerate(settings):
            pred = actual if setting_index == 0 else predict_from_refined(
                refined, base_route, a, e)
            classes[setting_index, image_index] = class_iou(pred, target)
            scores[setting_index, image_index] = classes[setting_index, image_index].mean()
        print(f"scored {image_index + 1}/{len(rows)}: {names[image_index]}", flush=True)
    baseline = float(scores[0].mean())
    reference = (json.loads(args.reference_json.read_text(encoding="utf-8"))
                 if args.reference_json else model349)
    archived = float(reference["raw_best_valid_image_mean_miou_empty0"])
    if not np.isclose(baseline, archived, rtol=0, atol=1e-10):
        raise AssertionError(f"fixed-route score parity failed: {baseline} vs {archived}")

    groups = {}
    for index, name in enumerate(names):
        match = re.match(r"(A\d+)_", name)
        if not match:
            raise ValueError(f"cannot parse specimen group: {name}")
        groups.setdefault(match.group(1), []).append(index)
    heldout = np.empty(len(names), dtype=np.int64)
    choices = []
    for group, indices in groups.items():
        training = [i for i in range(len(names)) if i not in indices]
        selected = int(np.argmax(scores[:, training].mean(axis=1)))
        heldout[indices] = selected
        choices.append({"heldout_group": group, "bias": settings[selected]})
    selected_scores = scores[heldout, np.arange(len(names))]
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": args.model_id,
        "metric": "20-image mean of four class IoUs; empty union=0",
        "baseline_fixed_route": baseline,
        "reference_json": str(args.reference_json) if args.reference_json else None,
        "fixed_spatial_blend": [base_route["gaussian_blend"],
                                base_route["eut_edge_blend"]],
        "gamma336": gamma336,
        "gamma347": gamma347,
        "class_order": ["Al3Ni", "Primary Si", "Al", "Eutectic Si"],
        "bias_order": ["Al", "Eutectic Si"],
        "raw_best_bias": settings[best],
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": classes[best].mean(axis=0).tolist(),
        "group_cv_heldout_miou_empty0": float(selected_scores.mean()),
        "group_cv_delta_pp": float((selected_scores - scores[0]).mean() * 100.0),
        "group_cv_choices": choices,
        "settings": [{"bias": setting, "score": float(score)} for setting, score
                     in zip(settings, scores.mean(axis=1))],
        "image_names": names,
        "baseline_image_scores": scores[0].tolist(),
        "raw_best_image_scores": scores[best].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_fixed_route", "raw_best_bias", "raw_best_valid_image_mean_miou_empty0",
        "group_cv_heldout_miou_empty0", "group_cv_delta_pp")},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
