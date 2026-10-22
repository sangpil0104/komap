#!/usr/bin/env python3
"""Fixed Model349 ablation of upstream base components on cached valid output.

No coefficient search. Each removed probability term is subtracted from the
pre-auxiliary mixture and the four channels are renormalized. This is a
diagnostic on already-used valid, not an independent test estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import model349_source_prune_audit as audit
from highres1024_search import MATRIX


def candidate_bases(item: dict, preaux: np.ndarray):
    params = item["params"]
    high_weight = float(params["high_weight"])
    pair_weight = float(params["pair_weight"])
    branch_weight = 1.0 - high_weight - pair_weight
    if min(high_weight, pair_weight, branch_weight) < 0:
        raise ValueError("negative base component coefficient")
    terms = [branch_weight * MATRIX[:, index, None, None]
             * item["branches"][index] for index in range(3)]
    pair_term = pair_weight * item["pair"]
    known = sum(terms) + pair_term
    scale = known.sum(axis=0, keepdims=True) + high_weight
    unnormalized = preaux * scale

    def remove(term):
        output = np.maximum(unnormalized - term, 1e-8)
        return output / np.maximum(output.sum(axis=0, keepdims=True), 1e-7)

    return {
        "full_349": preaux,
        "without_branch0": remove(terms[0]),
        "without_branch1": remove(terms[1]),
        "without_branch2": remove(terms[2]),
        "without_all_three_branches": remove(sum(terms)),
        "without_pair": remove(pair_term),
        "without_pair_and_all_branches": remove(known),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    outputs = args.root / "outputs"
    result = lambda name: json.loads((outputs / name / "results.json").read_text())
    route = result("model339_joint_gamma_route_search")["best_image_empty_zero"]
    gamma329 = tuple(result("model341_model329_joint_route_search")["best_image_empty_zero"]["gamma329"])
    gamma336 = tuple(result("model346_model336_into_341_route")["raw_best_gamma336"])
    result349 = result("model349_model347_into_346_route")
    gamma347 = tuple(result349["raw_best_gamma347"])
    fixed = next(audit.setting_variants(route, gamma329, gamma336, gamma347))
    rows = audit.load_rows(args.root)
    scores = {name: [] for name in candidate_bases(rows[0]["item"], rows[0]["base"])}
    for index, row in enumerate(rows, 1):
        guide = audit.model341.route.bilateral_weights(row["item"]["z"])
        target = row["item"]["target"].astype(np.int64)
        for name, base in candidate_bases(row["item"], row["base"]).items():
            changed = dict(row, base=base)
            pred = audit.prediction(changed, fixed, guide).astype(np.int64)
            matrix = np.bincount((target * 4 + pred).ravel(), minlength=16).reshape(4, 4)
            intersection = np.diag(matrix)
            union = matrix.sum(0) + matrix.sum(1) - intersection
            iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
            scores[name].append(float(iou.mean()))
        print(f"{index}/{len(rows)} {row['name']}", flush=True)
    expected = float(result349["raw_best_valid_image_mean_miou_empty0"])
    baseline = float(np.mean(scores["full_349"]))
    if abs(baseline - expected) > 1e-9:
        raise AssertionError(f"baseline mismatch {baseline} != {expected}")
    report = {
        "metric": "image-wise four-class mIoU, empty-union=0",
        "evaluation_role": "fixed ablations on already used valid; not independent selection",
        "baseline": baseline,
        "variants": {name: {"score": float(np.mean(values)),
                            "delta_percentage_points": float(100 * (np.mean(values) - baseline)),
                            "image_scores": values}
                     for name, values in scores.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: [round(data["score"], 9),
                              round(data["delta_percentage_points"], 4)]
                      for name, data in report["variants"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
