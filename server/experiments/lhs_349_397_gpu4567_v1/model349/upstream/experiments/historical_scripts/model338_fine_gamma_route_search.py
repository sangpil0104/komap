#!/usr/bin/env python3
"""Fine-grained probability-blend sweep around Model335's strict best route.

Keep each class's source expert fixed to Model335's strict-best choice and vary
one class's blend gamma at a time over 0.1..1.0. Score exact image-wise IoU
under both empty-union policies and leave-one-specimen-group-out route choice.
This is CPU-only; it loads cached probabilities and does no model inference.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import model335_exhaustive_b4_route_search as route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "outputs/model338_fine_gamma_route_search/results.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    previous = json.loads((root / "outputs/model335_exhaustive_b4_routes/results.json")
                          .read_text(encoding="utf-8"))
    parent = previous["best_image_empty_zero"]
    entry_path = root / "outputs/model183_metric_calibration/entries.pkl"
    entries = pickle.load(entry_path.open("rb"))
    source_paths = {
        "195": root / "outputs/model195_native_resume/valid_probs.npz",
        "217": root / "outputs/model220_native_tent_extended/valid_probs_06.npz",
        "269": root / "outputs/model269_fullstats_1x_extended/valid_probs.npz",
        "322": root / "outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz",
        "326": root / "outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz",
    }
    archives = {}
    names = None
    for source, path in source_paths.items():
        archive_names, values = route.load_archive(
            path, team_order=source in ("322", "326"))
        if names is None:
            names = archive_names
        elif set(archive_names) != set(names):
            raise ValueError(f"image key mismatch for Model{source}")
        archives[source] = values
    if names is None or len(entries) != len(names):
        raise ValueError("entry / validation probability count mismatch")

    half = {name: 0.5 * (archives["322"][name] + archives["326"][name])
            for name in names}
    archives["b4mean"] = half
    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        rows.append({
            "index": index,
            "item": item,
            "base": route.undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype("float32"),
                **{source: values[name] for source, values in archives.items()},
            },
        })

    fields = (
        "primary_source", "primary_gamma", "al3ni_source", "al3ni_gamma",
        "al_source", "al_gamma", "eutectic_source", "eutectic_gamma",
        "prior_scale", "beta_scale", "al_delta", "eut_delta",
        "gaussian_blend", "eut_edge_blend", "cleanup_threshold",
    )
    seed = {key: parent[key] for key in fields if key in parent}
    settings = []
    identities = set()
    reference = previous["baseline"]
    reference_setting = {key: reference[key] for key in fields if key in reference}
    reference_setting["stage"] = "model270_reference"
    reference_setting["candidate"] = reference["candidate"]
    settings.append(reference_setting)
    identities.add(tuple(reference_setting.get(key) for key in fields))
    gamma_fields = (
        ("Al3Ni", "al3ni_gamma"),
        ("Primary Si", "primary_gamma"),
        ("Al", "al_gamma"),
        ("Eutectic Si", "eutectic_gamma"),
    )
    for class_name, field in gamma_fields:
        for step in range(1, 11):
            gamma = step / 10.0
            setting = dict(seed)
            setting[field] = gamma
            identity = tuple(setting.get(key) for key in fields)
            if identity in identities:
                continue
            identities.add(identity)
            setting["stage"] = f"model338_{field}_{gamma:.1f}"
            setting["candidate"] = (
                f"Model335 strict-best experts; {class_name} gamma={gamma:.1f}")
            settings.append(setting)

    records, _image_confusion = route.evaluate(rows, settings, names, args.workers)
    strict_best = max(records, key=lambda item: item["image_mean_miou"])
    skip_best = max(records, key=lambda item: item["image_mean_miou_exclude"])
    cv_strict = route.grouped_cv(records, names, "image_scores", seed=338)
    cv_skip = route.grouped_cv(records, names, "image_scores_exclude", seed=339)
    result = {
        "model": 338,
        "metric": previous["metric"],
        "candidate_count": len(records),
        "parent_model335_strict": parent["image_mean_miou"],
        "parent_model335_nan_skip": previous["best_image_empty_skipped"][
            "image_mean_miou_exclude"],
        "best_image_empty_zero": strict_best,
        "best_image_empty_skipped": skip_best,
        "grouped_cv_image_empty_zero": cv_strict,
        "grouped_cv_image_empty_skipped": cv_skip,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(records),
        "parent_strict": result["parent_model335_strict"],
        "strict_best": {
            "score": strict_best["image_mean_miou"],
            "exclude": strict_best["image_mean_miou_exclude"],
            "candidate": strict_best["candidate"],
        },
        "skip_best": {
            "score": skip_best["image_mean_miou"],
            "exclude": skip_best["image_mean_miou_exclude"],
            "candidate": skip_best["candidate"],
        },
        "grouped_cv_strict_delta_pp": cv_strict["image_mean_delta_pp"],
        "grouped_cv_strict_95ci_delta_pp": cv_strict[
            "group_bootstrap_95ci_delta_pp"],
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
