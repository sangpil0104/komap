#!/usr/bin/env python3
"""Test Model362-selected Model361 weights on fixed Model360 inference.

This is a same-validation diagnostic, not independent route selection: the
Model362 gamma came from these validation images. The zero-expert candidate
must reproduce Model360 pixel masks and its archived official empty=0 score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import model355_model349_spatial_audit as spatial
from model360_entropy_gated_d4_into_355 import entropy


def predict_with_expert(row, base_route, gamma329, gamma336, gamma347,
                        gamma361, g, e):
    route = spatial.audit.model350.model341.route
    mixed = route.make_mixed(row, base_route)
    for key, gammas in (("329", gamma329), ("336", gamma336),
                        ("347", gamma347), ("361", gamma361)):
        for class_id, gamma in enumerate(gammas):
            if gamma:
                mixed = route.replace_classes(mixed, row["sources"][key],
                                              (class_id,), gamma)
    item = row["item"]
    base_prior, delta = spatial.audit.prior_terms(item, mixed)
    prior = base_route["prior_scale"] * base_prior
    prior[0] += (base_route["prior_scale"] * base_route["beta_scale"]
                 * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + base_route["al_delta"]
    logits[3] += item["params"]["eut_bias"] + base_route["eut_delta"]
    weights = route.bilateral_weights(item["z"])
    refined = logits.copy()
    refined[1] = (g * gaussian_filter(logits[1], sigma=0.50, mode="reflect")
                  + (1.0 - g) * spatial.audit.apply_bilateral(logits[1], weights))
    refined[3] = ((1.0 - e) * logits[3]
                  + e * spatial.audit.apply_bilateral(logits[3], weights))
    return spatial.audit.predict_from_refined(refined, base_route, 0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    outputs = args.root / "outputs"
    rows, names, base_route, gamma329, _ = spatial.audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    load_archive = spatial.audit.model350.model341.route.load_archive
    archives = {}
    for key, path in (
        ("347", outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"),
        ("351", outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"),
        ("361", outputs / "model361_hrnet_scale3_finetune/valid_eval/valid_probabilities.npz"),
    ):
        _, archives[key] = load_archive(path, team_order=True)
        if set(archives[key]) != set(names):
            raise ValueError(f"Model{key} image-key mismatch")
    model346 = json.loads((outputs / "model346_model336_into_341_route/results.json")
                           .read_text(encoding="utf-8"))
    model349 = json.loads((outputs / "model349_model347_into_346_route/results.json")
                           .read_text(encoding="utf-8"))
    model355 = json.loads((outputs / "model355_model349_spatial_audit/results.json")
                           .read_text(encoding="utf-8"))
    model360 = json.loads((outputs / "model360_entropy_gated_d4_into_355/results.json")
                           .read_text(encoding="utf-8"))
    model362 = json.loads((outputs / "model362_model361_into_349_route/results.json")
                           .read_text(encoding="utf-8"))
    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    g, e = model355["raw_best_spatial_blend"]
    margin, strength = model360["raw_best_entropy_gate"]
    fixed_route = dict(base_route, gaussian_blend=g, eut_edge_blend=e)
    zero = (0.0, 0.0, 0.0, 0.0)
    settings = [zero]
    for key in ("raw_best_gamma361", "majority_group_cv_gamma361"):
        gamma = tuple(model362[key])
        if gamma not in settings:
            settings.append(gamma)
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    classes = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    for image_index, (name, row) in enumerate(zip(names, rows)):
        p336 = row["sources"]["336"]
        p351 = archives["351"][name]
        gate = (entropy(p351) + margin < entropy(p336)).astype(np.float32)
        row["sources"]["336"] = p336 + strength * gate[None] * (p351 - p336)
        row["sources"]["347"] = archives["347"][name]
        row["sources"]["361"] = archives["361"][name]
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        expected = spatial.prediction(parts, fixed_route, g, e)
        target = row["item"]["target"]
        for setting_index, gamma in enumerate(settings):
            pred = predict_with_expert(row, base_route, gamma329, gamma336,
                                       gamma347, gamma, g, e)
            if setting_index == 0 and not np.array_equal(pred, expected):
                raise AssertionError(f"Model360 pixel parity failed for {name}")
            classes[setting_index, image_index] = spatial.audit.class_iou(pred, target)
            scores[setting_index, image_index] = classes[setting_index, image_index].mean()
        row["sources"]["336"] = p336
        print(f"scored {image_index + 1}/{len(rows)}: {name}", flush=True)
    baseline = float(scores[0].mean())
    archived = float(model360["raw_best_valid_image_mean_miou_empty0"])
    if not np.isclose(baseline, archived, atol=1e-10, rtol=0):
        raise AssertionError(f"Model360 score parity failed: {baseline} vs {archived}")

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
        choices.append({"heldout_group": group, "selected_gamma361": settings[best]})
    held_scores = scores[selected, np.arange(len(names))]
    best = int(np.argmax(scores.mean(axis=1)))
    result = {
        "model": 363,
        "metric": "20-image mean of four class IoUs; empty union=0",
        "baseline_model360": baseline,
        "model362_gamma_raw": model362["raw_best_gamma361"],
        "model362_gamma_majority": model362["majority_group_cv_gamma361"],
        "raw_best_gamma361_among_imported": settings[best],
        "raw_best_valid_image_mean_miou_empty0": float(scores[best].mean()),
        "raw_best_class_mean_iou_empty0": classes[best].mean(axis=0).tolist(),
        "group_cv_heldout_miou_empty0": float(held_scores.mean()),
        "group_cv_delta_pp": float((held_scores - scores[0]).mean() * 100.0),
        "group_cv_choices": choices,
        "settings": [{"gamma361": gamma, "score": float(score)}
                     for gamma, score in zip(settings, scores.mean(axis=1))],
        "image_names": names,
        "baseline_image_scores": scores[0].tolist(),
        "raw_best_image_scores": scores[best].tolist(),
        "warning": "Gamma chosen using same valid images in Model362; held-out choice here is diagnostic only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model360", "raw_best_gamma361_among_imported",
        "raw_best_valid_image_mean_miou_empty0", "group_cv_heldout_miou_empty0",
        "group_cv_delta_pp")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
