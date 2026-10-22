#!/usr/bin/env python3
"""CPU-only class-wise routing of the Model336 HRNet branch into Model270.

Enumerates 5**4 routes: retain Model270 for a class, or blend Model336 at
gamma 0.25/0.50/0.75/1.00. Uses Model335's exact calibration/spatial route
implementation, and verifies probability/image/label alignment before scoring.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

import model335_exhaustive_b4_route_search as route
from model1_unet_resnet34 import decode_mask
from model323_model270_branch_route_probe import canonical_image_key, grouped_cv
from model198_fast_branch_search import default_setting


FIELDS = (
    (0, "Al3Ni", "al3ni_source", "al3ni_gamma"),
    (1, "Primary Si", "primary_source", "primary_gamma"),
    (2, "Al", "al_source", "al_gamma"),
    (3, "Eutectic Si", "eutectic_source", "eutectic_gamma"),
)
GAMMAS = (0.25, 0.50, 0.75, 1.00)


def build_settings() -> list[dict]:
    import itertools

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
    choices = [(None, 0.0, "Model270")]
    choices.extend(("336", gamma, f"336@{gamma:.2f}") for gamma in GAMMAS)
    settings = []
    for index, combination in enumerate(itertools.product(choices, repeat=4)):
        config = dict(reference)
        labels = []
        for (_, name, source_key, gamma_key), (source, gamma, label) in zip(FIELDS, combination):
            labels.append(f"{name}={label}")
            if source is not None:
                config[source_key] = source
                config[gamma_key] = gamma
            elif source_key == "al_source":
                config.pop("al_source", None)
                config["al_gamma"] = 0.0
        config["stage"] = "model270_reference" if index == 0 else f"hrnet_route_{index:04d}"
        config["candidate"] = " | ".join(labels)
        settings.append(config)
    if len(settings) != 625:
        raise AssertionError(f"expected 625 routes; found {len(settings)}")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True, type=Path)
    parser.add_argument("--model195", required=True, type=Path)
    parser.add_argument("--tent20", required=True, type=Path)
    parser.add_argument("--model269", required=True, type=Path)
    parser.add_argument("--model336", required=True, type=Path,
                        help="team-order [Al, Al3Ni, Primary Si, Eutectic Si] probabilities")
    parser.add_argument("--valid-masks", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.entries.open("rb") as handle:
        entries = pickle.load(handle)
    source_paths = {
        "195": (args.model195, False),
        "217": (args.tent20, False),
        "269": (args.model269, False),
        "336": (args.model336, True),
    }
    archives: dict[str, dict[str, np.ndarray]] = {}
    archive_names: list[str] | None = None
    for source, (path, team_order) in source_paths.items():
        current_names, values = route.load_archive(path, team_order=team_order)
        if archive_names is None:
            archive_names = current_names
        elif set(current_names) != set(archive_names):
            raise ValueError(f"image-key mismatch for source {source}")
        archives[source] = values
    assert archive_names is not None
    if len(entries) != len(archive_names):
        raise ValueError(f"entry count {len(entries)} != probability count {len(archive_names)}")

    mask_paths = sorted(args.valid_masks.glob("*_mask.png"))
    mask_names = [canonical_image_key(path.name[:-len("_mask.png")]) for path in mask_paths]
    if len(set(mask_names)) != len(mask_names) or set(mask_names) != set(archive_names):
        raise ValueError("valid-mask IDs do not match cached entries/probability IDs")
    # Model183 cached entries are created in sorted mask order but do not store
    # IDs; target equality below proves this order before any maps are attached.
    names = mask_names

    rows = []
    for index, (name, item, mask_path) in enumerate(zip(names, entries, mask_paths)):
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        if not np.array_equal(item["target"], target):
            raise ValueError(f"cached target does not match {mask_path.name}; refusing route search")
        expected_shape = item["target"].shape
        for source, values in archives.items():
            value = values[name]
            if value.shape != (4, *expected_shape):
                raise ValueError(f"{source} probability shape mismatch for {name}: {value.shape}")
            if not np.allclose(value.sum(axis=0), 1.0, atol=3e-3):
                raise ValueError(f"{source} probability normalization failed for {name}")
        rows.append({
            "index": index,
            "item": item,
            "base": route.undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: values[name] for source, values in archives.items()},
            },
        })

    settings = build_settings()
    records, image_confusion = route.evaluate(rows, settings, names, args.workers)
    result = {
        "metric": "Exact per-image mean of four class IoUs; empty union=0 and NaN-skip",
        "route_count": len(records),
        "source_options": ["keep Model270", "Model336 at gamma 0.25/0.50/0.75/1.00"],
        "class_order": ["Al3Ni", "Primary Si", "Al", "Eutectic Si"],
        "candidate_probability_input_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
        "baseline": next(row for row in records if row["stage"] == "model270_reference"),
        "best_image_empty_zero": max(records, key=lambda row: row["image_mean_miou"]),
        "best_image_empty_skipped": max(records, key=lambda row: row["image_mean_miou_exclude"]),
        "best_image_compat_diagnostic_only": max(records, key=lambda row: row["image_miou_compat"]),
        "grouped_cv_image_empty_zero": grouped_cv(records, names, "image_scores", 337),
        "grouped_cv_image_empty_skipped": grouped_cv(records, names, "image_scores_exclude", 338),
        "grouped_cv_disabled_diagnostic_only": route.grouped_cv_disabled(records, image_confusion, names),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "route_count", "baseline", "best_image_empty_zero",
        "best_image_empty_skipped", "grouped_cv_image_empty_zero",
        "grouped_cv_image_empty_skipped")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
