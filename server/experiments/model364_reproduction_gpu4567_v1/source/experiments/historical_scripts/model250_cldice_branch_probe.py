#!/usr/bin/env python3
"""Exact rare-class fusion probe for the eutectic clDice branch."""

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
    archives = {
        "195": np.load(args.model195),
        "217": np.load(args.tent20),
        "244": np.load(args.model244),
        "249": np.load(args.model249),
    }
    key_orders = {tuple(archive.files) for archive in archives.values()}
    if len(key_orders) != 1:
        raise ValueError("probability archives differ in image keys/order")
    values = {name: [archive[key].astype(np.float32) for key in archive.files]
              for name, archive in archives.items()}
    if any(len(items) != len(entries) for items in values.values()):
        raise ValueError("probability archive and calibration cache lengths differ")

    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "index": index,
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{name: items[index] for name, items in values.items()}},
        })

    settings = [default_setting(
        stage="model250_cldice_rare_class_fusion",
        primary_source="195", primary_gamma=0.75,
        al3ni_source="217", al3ni_gamma=0.40,
        eutectic_source="217", eutectic_gamma=0.40,
        prior_scale=1.25, beta_scale=0.80,
        al_delta=0.0, eut_delta=0.15,
        gaussian_blend=0.50, eut_edge_blend=0.25,
        cleanup_threshold=128,
    )]
    sources = ("217", "244", "249")
    for primary_gamma in (0.75, 0.85):
        for al_source in sources:
            for al_gamma in (0.20, 0.40):
                for eut_source in sources:
                    for eut_gamma in (0.20, 0.40):
                        settings.append(default_setting(
                            stage="model250_cldice_rare_class_fusion",
                            primary_source="195", primary_gamma=primary_gamma,
                            al3ni_source=al_source, al3ni_gamma=al_gamma,
                            eutectic_source=eut_source, eutectic_gamma=eut_gamma,
                            prior_scale=1.25, beta_scale=0.80,
                            al_delta=0.0, eut_delta=0.15,
                            gaussian_blend=0.50, eut_edge_blend=0.25,
                            cleanup_threshold=128,
                        ))
    records = evaluate(rows, settings, exact=True)
    result = {
        "probe": "model249 eutectic clDice source with TENT20/model244 in class-wise exact fusion",
        "record_count": len(records),
        "best_image": max(records, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(records, key=lambda row: row["image_mean_miou_exclude"]),
        "best_image_compat": max(records, key=lambda row: row["image_miou_compat"]),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_image_compat"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
