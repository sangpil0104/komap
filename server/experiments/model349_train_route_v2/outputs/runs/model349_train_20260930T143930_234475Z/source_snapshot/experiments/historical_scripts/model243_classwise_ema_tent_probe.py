#!/usr/bin/env python3
"""Exact class-wise route probe for model242 EMA and 20-step TENT."""

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
    archives = {"195": np.load(args.model195), "217": np.load(args.model217),
                "242": np.load(args.model242)}
    keys = {name: list(archive.files) for name, archive in archives.items()}
    if len(set(tuple(value) for value in keys.values())) != 1:
        raise ValueError("probability archive keys/order differ")
    values = {name: [archive[key].astype(np.float32) for key in archive.files]
              for name, archive in archives.items()}
    if any(len(items) != len(entries) for items in values.values()):
        raise ValueError("probability archives and calibration entries differ in length")

    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "index": index, "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{name: items[index] for name, items in values.items()}},
        })

    settings = []
    for primary_gamma in (0.75, 0.85):
        for al_source in ("217", "242"):
            for al_gamma in (0.20, 0.40):
                for eut_source in ("217", "242"):
                    for eut_gamma in (0.20, 0.40):
                        settings.append(default_setting(
                            stage="model243_classwise_ema_tent",
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
        "probe": "32 exact candidates; independent Al3Ni/Eutectic sources from TENT and EMA stem",
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
    parser.add_argument("--model217", required=True)
    parser.add_argument("--model242", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
