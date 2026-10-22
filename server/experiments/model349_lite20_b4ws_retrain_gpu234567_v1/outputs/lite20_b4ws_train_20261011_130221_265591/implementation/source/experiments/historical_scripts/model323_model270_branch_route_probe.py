#!/usr/bin/env python3
"""Probe a held-out validation-group fusion on top of the current Model270 route.

This complements Model322's train-only OOF comparison against the weaker
Model303 parent: it checks whether a candidate branch can improve the deployed
high-scoring Model270 route. Route selection is leave-one-specimen-group-out.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import itertools
import json
import pickle
import re
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import (
    bilateral_weights,
    default_setting,
    make_mixed,
    predict,
)


TEAM_TO_MAIN = (1, 2, 0, 3)  # [Al, Al3Ni, Primary, Eutectic] -> main order
MIXABLE_CLASSES = {
    0: ("Al3Ni", "al3ni_source", "al3ni_gamma"),
    1: ("Primary Si", "primary_source", "primary_gamma"),
    2: ("Al", "al_source", "al_gamma"),
    3: ("Eutectic Si", "eutectic_source", "eutectic_gamma"),
}


def load_archive(path: Path, *, team_order: bool = False):
    archive = np.load(path)
    values = {}
    for key in archive.files:
        value = archive[key].astype(np.float32)
        if team_order:
            if value.ndim != 3 or value.shape[0] != 4:
                raise ValueError(f"expected 4xHxW B4 probability for {key}: {value.shape}")
            value = value[list(TEAM_TO_MAIN)]
        values[key] = value
    return archive.files, values


def canonical_image_key(key: str) -> str:
    """Normalize archive keys from either `sample` or `sample_image` format."""
    return key[:-6] if key.endswith("_image") else key


def specimen_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def evaluate_candidates(rows: list[dict], settings: list[tuple[str, dict]],
                        candidate_id: str) -> list[dict]:
    """Exact spatial route evaluation, including optional Al branch blending."""
    count = len(settings)
    confusion = np.zeros((count, 4, 4), dtype=np.int64)
    image_scores = np.zeros((count, len(rows)), dtype=np.float64)
    image_scores_exclude = np.zeros_like(image_scores)
    for row in rows:
        guide_weights = bilateral_weights(row["item"]["z"])
        target = row["item"]["target"]
        for index, (_, setting) in enumerate(_local_progress(settings, 'model323_model270_branch_route_probe')):
            mixed = make_mixed(row, setting)
            al_gamma = float(setting.get("al_gamma", 0.0))
            if al_gamma:
                al = row["sources"][candidate_id][2]
                mixed[2] = (1.0 - al_gamma) * mixed[2] + al_gamma * al
                mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
            prediction = predict(row["item"], mixed, setting, guide_weights, exact=True)
            matrix = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1),
                                 minlength=16).reshape(4, 4)
            confusion[index] += matrix
            intersection = np.diag(matrix).astype(np.float64)
            union = matrix.sum(0) + matrix.sum(1) - intersection
            iou = intersection / np.maximum(union, 1.0)
            image_scores[index, row["index"]] = iou.mean()
            image_scores_exclude[index, row["index"]] = iou[union > 0].mean()

    records = []
    for index, (name, setting) in enumerate(_local_progress(settings, 'model323_model270_branch_route_probe')):
        matrix = confusion[index]
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        image_class_iou = intersection / np.maximum(union, 1.0)
        records.append({
            **setting,
            "candidate": name,
            "image_miou_compat": float(image_class_iou.mean()),
            "image_class_iou": {key: float(value) for key, value in zip(
                ("Al3Ni", "Primary Si", "Al", "Eutectic Si"), image_class_iou)},
            "image_mean_miou": float(image_scores[index].mean()),
            "image_mean_miou_exclude": float(image_scores_exclude[index].mean()),
            "image_scores": image_scores[index].tolist(),
            "image_scores_exclude": image_scores_exclude[index].tolist(),
            "confusion": matrix.tolist(),
        })
    return records


def grouped_cv(records: list[dict], names: tuple[str, ...], score_key: str,
               seed: int) -> dict:
    reference = next(row for row in records if row["stage"] == "model270_reference")
    reference_scores = np.asarray(reference[score_key], dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(specimen_group(name), []).append(index)

    delta = np.zeros(len(names), dtype=np.float64)
    fold_choices = []
    for held_group, held_indices in groups.items():
        train_indices = [i for group, indices in groups.items() if group != held_group
                         for i in indices]
        selected = max(records, key=lambda row: float(
            np.asarray(row[score_key])[train_indices].mean()))
        selected_scores = np.asarray(selected[score_key], dtype=np.float64)
        delta[held_indices] = selected_scores[held_indices] - reference_scores[held_indices]
        fold_choices.append({"heldout_specimen": held_group,
                             "selected": selected["candidate"]})

    rng = np.random.default_rng(seed)
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [i for group in sampled for i in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0
    return {
        "image_mean_delta_pp": float(delta.mean() * 100.0),
        "positive_image_count": int((delta > 0).sum()),
        "image_count": len(names),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "fold_choices": fold_choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True, type=Path)
    parser.add_argument("--model195", required=True, type=Path)
    parser.add_argument("--tent20", required=True, type=Path)
    parser.add_argument("--model269", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-order", choices=("main", "team"), default="main")
    parser.add_argument("--classwise-grid", default="",
                        help="optional comma-separated gamma grid for a classwise Al3Ni/Primary Si/Al sweep")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.entries.open("rb") as handle:
        entries = pickle.load(handle)
    paths = {"195": args.model195, "217": args.tent20, "269": args.model269,
             args.candidate_id: args.candidate}
    archives = {}
    image_names = None
    for source, path in paths.items():
        team_order = source == args.candidate_id and args.candidate_order == "team"
        names, values = load_archive(path, team_order=team_order)
        normalized = {canonical_image_key(name): values[name] for name in names}
        if len(normalized) != len(names):
            raise ValueError(f"image-key normalization collision in {path}")
        if image_names is None:
            image_names = [canonical_image_key(name) for name in names]
        elif set(normalized) != set(image_names):
            missing = sorted(set(image_names) - set(normalized))
            extra = sorted(set(normalized) - set(image_names))
            raise ValueError(f"image-key set mismatch in {path}: missing={missing}, extra={extra}")
        archives[source] = normalized
    assert image_names is not None
    if len(entries) != len(image_names):
        raise ValueError(f"expected {len(entries)} cached images, found {len(image_names)}")

    rows = [{
        "index": index,
        "item": item,
        "base": undo_model181_primary_blend(item),
        "sources": {"base": None, "181": item["extra"].astype(np.float32),
                    **{source: archives[source][image_names[index]] for source in archives}},
    } for index, item in enumerate(entries)]

    reference = default_setting(
        stage="model270_reference",
        primary_source="195", primary_gamma=0.75,
        al3ni_source="217", al3ni_gamma=0.40,
        eutectic_source="269", eutectic_gamma=0.50,
        prior_scale=1.25, beta_scale=0.80,
        al_delta=0.0, eut_delta=0.15,
        gaussian_blend=0.50, eut_edge_blend=0.25,
        cleanup_threshold=128,
    )
    settings = [("reference", reference)]
    for subset in range(1, 1 << len(MIXABLE_CLASSES)):
        for gamma in (0.30, 0.50):
            config = dict(reference)
            config["stage"] = f"candidate_{args.candidate_id}_{subset}_{gamma:.2f}"
            candidate_classes = []
            for bit, (class_id, (class_name, source_key, gamma_key)) in enumerate(
                    MIXABLE_CLASSES.items()):
                if subset & (1 << bit):
                    config[source_key] = args.candidate_id
                    config[gamma_key] = gamma
                    candidate_classes.append(class_name)
            settings.append((f"classes={candidate_classes};gamma={gamma:.2f}", config))

    if args.classwise_grid:
        gamma_grid = [float(value) for value in args.classwise_grid.split(",")]
        if not gamma_grid or any(value < 0.0 or value > 1.0 for value in gamma_grid):
            raise ValueError("classwise gamma values must be in [0, 1]")
        # Model325's grouped-LOO mostly selected B4 fusion on these three
        # classes while retaining the existing Eutectic branch. Refine their
        # weights independently; do not tune Eutectic from the weak B4 branch.
        for gammas in itertools.product(gamma_grid, repeat=3):
            config = dict(reference)
            candidate_classes = []
            for class_id, gamma in zip((0, 1, 2), gammas):
                class_name, source_key, gamma_key = MIXABLE_CLASSES[class_id]
                config[source_key] = args.candidate_id
                config[gamma_key] = gamma
                candidate_classes.append(f"{class_name}:{gamma:.2f}")
            config["stage"] = "candidate_" + args.candidate_id + "_classwise_" + "_".join(
                f"{gamma:.2f}" for gamma in gammas)
            settings.append((f"classwise={candidate_classes}", config))

    # Also test the candidate as a full replacement for the three mixable classes.
    full = dict(reference)
    full.update(stage=f"candidate_{args.candidate_id}_full", primary_source=args.candidate_id,
                primary_gamma=1.0, al3ni_source=args.candidate_id, al3ni_gamma=1.0,
                al_source=args.candidate_id, al_gamma=1.0,
                eutectic_source=args.candidate_id, eutectic_gamma=1.0)
    settings.append(("all_mixable_classes=full_candidate", full))
    records = evaluate_candidates(rows, settings, args.candidate_id)

    result = {
        "metric": "per-image mean of 4 class IoUs; empty union=0, also report NaN-skip",
        "baseline": next(row for row in records if row["stage"] == "model270_reference"),
        "best_image_empty_zero": max(records, key=lambda row: row["image_mean_miou"]),
        "best_image_empty_skipped": max(records,
                                         key=lambda row: row["image_mean_miou_exclude"]),
        "grouped_cv_vs_model270_empty_zero": grouped_cv(
            records, image_names, "image_scores", 323),
        "grouped_cv_vs_model270_empty_skipped": grouped_cv(
            records, image_names, "image_scores_exclude", 324),
        "candidate_id": args.candidate_id,
        "candidate_order_input": args.candidate_order,
        "candidate_count": len(records),
        "mixable_classes": list(MIXABLE_CLASSES.values()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline", "best_image_empty_zero", "best_image_empty_skipped",
        "grouped_cv_vs_model270_empty_zero", "grouped_cv_vs_model270_empty_skipped")},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
