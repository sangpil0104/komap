#!/usr/bin/env python3
"""Confidence-gated pixelwise fusion of a high-resolution branch.

The high-resolution model is allowed to affect a selected phase only at pixels
where its class probability exceeds the established model225 route by a chosen
margin. Candidate selection is evaluated with leave-one-specimen-group-out CV.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from model181_raw_zcdf_aspp import score
from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, default_setting, make_mixed, predict


def metrics(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    archives = {
        "195": np.load(args.model195),
        "217": np.load(args.tent20),
        "265": np.load(args.highres),
    }
    if len({tuple(archive.files) for archive in archives.values()}) != 1:
        raise ValueError("probability archives differ in image keys/order")
    names = tuple(archives["265"].files)
    if len(names) != len(entries):
        raise ValueError("probability archive and route cache sizes differ")
    probs = {source: [archive[key].astype(np.float32) for key in archive.files]
             for source, archive in archives.items()}
    rows = [{
        "index": index,
        "item": item,
        "base": undo_model181_primary_blend(item),
        "sources": {
            "base": None,
            "181": item["extra"].astype(np.float32),
            **{source: values[index] for source, values in probs.items()},
        },
    } for index, item in enumerate(entries)]

    reference = default_setting(
        stage="model225_reference", primary_source="195", primary_gamma=0.75,
        al3ni_source="217", al3ni_gamma=0.40,
        eutectic_source="217", eutectic_gamma=0.40,
        prior_scale=1.25, beta_scale=0.80, al_delta=0.0, eut_delta=0.15,
        gaussian_blend=0.50, eut_edge_blend=0.25, cleanup_threshold=128,
    )
    class_sets = {
        "Al3Ni": (0,), "Primary Si": (1,), "Eutectic Si": (3,),
        "Al3Ni+Primary Si": (0, 1), "Al3Ni+Eutectic Si": (0, 3),
        "Primary Si+Eutectic Si": (1, 3),
        "Al3Ni+Primary Si+Eutectic Si": (0, 1, 3),
    }
    candidates = [{"name": "model225_reference", "classes": (),
                   "margin": 0.0, "gamma": 0.0}]
    for name, class_ids in class_sets.items():
        for margin in args.margins:
            for gamma in args.gammas:
                candidates.append({"name": name, "classes": class_ids,
                                   "margin": margin, "gamma": gamma})

    # Bilateral weights depend only on the fixed grayscale guide; compute them
    # once per image rather than repeating the neighborhood expansion for each
    # candidate in the sweep.
    weight_cache = [bilateral_weights(row["item"]["z"]) for row in rows]
    records = []
    for candidate in candidates:
        predictions, targets = [], []
        for row_index, row in enumerate(rows):
            base = make_mixed(row, reference)
            mixed = base.copy()
            if candidate["classes"]:
                highres = row["sources"]["265"]
                for class_id in candidate["classes"]:
                    gate = highres[class_id] > base[class_id] + candidate["margin"]
                    if gate.any():
                        mixed[class_id, gate] = (
                            (1.0 - candidate["gamma"]) * base[class_id, gate]
                            + candidate["gamma"] * highres[class_id, gate]
                        )
                mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
            weights = weight_cache[row_index]
            pred = predict(row["item"], mixed, reference, weights, exact=True)
            predictions.append(pred)
            targets.append(row["item"]["target"])
        records.append({**candidate, **metrics(predictions, targets)})

    ref_scores = np.asarray(records[0]["image_scores"], dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        match = re.match(r"(A\d+)_", Path(name).name)
        group = match.group(1) if match else Path(name).stem.split("_")[0]
        groups.setdefault(group, []).append(index)
    cv_delta = np.zeros(len(entries), dtype=np.float64)
    fold_choices = []
    for heldout, held_indices in groups.items():
        train_indices = [i for group, ids in groups.items() if group != heldout for i in ids]
        selected = max(records, key=lambda r: float(np.mean(
            np.asarray(r["image_scores"])[train_indices])))
        selected_scores = np.asarray(selected["image_scores"])
        cv_delta[held_indices] = selected_scores[held_indices] - ref_scores[held_indices]
        fold_choices.append({"heldout_specimen": heldout, "selected": selected["name"],
                             "margin": selected["margin"], "gamma": selected["gamma"],
                             "classes": list(selected["classes"])})

    rng = np.random.default_rng(271)
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        ids = [i for group in sampled for i in groups[str(group)]]
        bootstrap[draw] = float(cv_delta[ids].mean())

    cv = {
        "image_mean_delta_pp": float(cv_delta.mean() * 100.0),
        "positive_image_count": int((cv_delta > 0).sum()),
        "heldout_image_count": len(entries),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(value * 100.0) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "fold_choices": fold_choices,
    }
    result = {
        "method": "pixelwise confidence-gated high-resolution probability fusion",
        "candidate_count": len(candidates),
        "baseline": records[0],
        "best_image": max(records, key=lambda r: r["image_mean_miou"]),
        "best_image_compat": max(records, key=lambda r: r["image_miou_compat"]),
        "grouped_cv_vs_model225": cv,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_image_compat", "grouped_cv_vs_model225"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--highres", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margins", nargs="+", type=float, default=(0.0, 0.05, 0.10))
    parser.add_argument("--gammas", nargs="+", type=float, default=(0.25, 0.50))
    main(parser.parse_args())
