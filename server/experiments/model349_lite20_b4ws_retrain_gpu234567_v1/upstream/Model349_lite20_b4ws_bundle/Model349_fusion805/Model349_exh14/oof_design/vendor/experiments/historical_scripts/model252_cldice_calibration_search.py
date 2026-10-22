#!/usr/bin/env python3
"""Focused calibration of eutectic clDice probabilities in the best route."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import default_setting, evaluate


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    archives = {"195": np.load(args.model195), "217": np.load(args.tent20),
                "244": np.load(args.model244), "249": np.load(args.model249)}
    key_orders = {tuple(archive.files) for archive in archives.values()}
    if len(key_orders) != 1:
        raise ValueError("probability archive image keys/orders differ")
    source_values = {name: [archive[key].astype(np.float32) for key in archive.files]
                     for name, archive in archives.items()}
    if any(len(values) != len(entries) for values in source_values.values()):
        raise ValueError("probability archive and route cache sizes differ")
    rows = [{
        "index": index,
        "item": item,
        "base": undo_model181_primary_blend(item),
        "sources": {"base": None, "181": item["extra"].astype(np.float32),
                    **{name: values[index] for name, values in source_values.items()}},
    } for index, item in enumerate(entries)]

    settings = []
    for eut_source in ("217", "244", "249"):
        for eut_gamma in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40):
            for eut_delta in (-0.45, -0.30, -0.15, 0.0, 0.15):
                settings.append(default_setting(
                    stage="model252_cldice_calibration",
                    primary_source="195", primary_gamma=0.75,
                    al3ni_source="217", al3ni_gamma=0.40,
                    eutectic_source=eut_source, eutectic_gamma=eut_gamma,
                    prior_scale=1.25, beta_scale=0.80,
                    al_delta=0.0, eut_delta=eut_delta,
                    gaussian_blend=0.50, eut_edge_blend=0.25,
                    cleanup_threshold=128,
                ))
    # Cheap logits pass ranks candidates; exact spatial filtering is then run
    # on the strongest candidates plus the unchanged model225 reference.
    cheap = evaluate(rows, settings, exact=False)
    ordered = sorted(cheap, key=lambda row: row["image_mean_miou"], reverse=True)
    exact_settings = []
    seen = set()
    for row in ordered[:args.exact_top]:
        setting = {key: row[key] for key in settings[0]}
        identity = tuple(setting[key] for key in sorted(setting))
        if identity not in seen:
            seen.add(identity)
            exact_settings.append(setting)
    reference = default_setting(
        stage="model252_model225_reference",
        primary_source="195", primary_gamma=0.75,
        al3ni_source="217", al3ni_gamma=0.40,
        eutectic_source="217", eutectic_gamma=0.40,
        prior_scale=1.25, beta_scale=0.80,
        al_delta=0.0, eut_delta=0.15,
        gaussian_blend=0.50, eut_edge_blend=0.25,
        cleanup_threshold=128,
    )
    ref_id = tuple(reference[key] for key in sorted(reference))
    if ref_id not in seen:
        exact_settings.append(reference)
    exact = evaluate(rows, exact_settings, exact=True)
    result = {
        "probe": "120 eutectic-source/gamma/bias cheap settings; exact top candidates plus model225 reference",
        "cheap_count": len(cheap),
        "exact_count": len(exact),
        "best_image": max(exact, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(exact, key=lambda row: row["image_mean_miou_exclude"]),
        "best_global": max(exact, key=lambda row: row["global_miou"]),
        "top_cheap": [{key: row[key] for key in (
            "image_mean_miou", "eutectic_source", "eutectic_gamma", "eut_delta")}
            for row in ordered[:20]],
        "records": exact,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_global"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-top", type=int, default=24)
    main(parser.parse_args())
