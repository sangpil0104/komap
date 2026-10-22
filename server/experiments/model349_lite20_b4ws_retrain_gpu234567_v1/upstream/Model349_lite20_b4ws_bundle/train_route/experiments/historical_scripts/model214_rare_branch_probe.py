#!/usr/bin/env python3
"""Probe new branches only on rare classes while keeping model195 Primary."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import default_setting, evaluate


def main(args):
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    paths = {
        "195": args.model195,
        "207": args.model207,
        "209": args.model209,
        "211": args.model211,
    }
    values = {}
    for key, path in paths.items():
        archive = np.load(path)
        values[key] = [archive[name].astype(np.float32) for name in archive.files]
    rows = []
    for index, item in enumerate(entries):
        rows.append({"index": index, "item": item,
                     "base": undo_model181_primary_blend(item),
                     "sources": {"base": None, "181": item["extra"].astype(np.float32),
                                 **{key: array[index] for key, array in values.items()}}})
    settings = []
    for primary_gamma in (0.70, 0.75, 0.80):
        base = default_setting(primary_source="195", primary_gamma=primary_gamma,
                               al3ni_source="195", al3ni_gamma=0.075,
                               eutectic_source="195", eutectic_gamma=0.075)
        settings.append(base)
        for class_name in ("al3ni", "eutectic"):
            for source in ("207", "209", "211"):
                for gamma in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
                    candidate = dict(base)
                    candidate["stage"] = f"{class_name}_{source}"
                    candidate[f"{class_name}_source"] = source
                    candidate[f"{class_name}_gamma"] = gamma
                    settings.append(candidate)
        for source in ("207", "209", "211"):
            for gamma in (0.025, 0.05, 0.075, 0.10, 0.15):
                candidate = dict(base)
                candidate["stage"] = f"joint_{source}"
                candidate["al3ni_source"] = source
                candidate["eutectic_source"] = source
                candidate["al3ni_gamma"] = gamma
                candidate["eutectic_gamma"] = gamma
                settings.append(candidate)
    records = evaluate(rows, settings, exact=True)
    result = {"record_count": len(records),
              "best_image": max(records, key=lambda x: x["image_mean_miou"]),
              "best_exclude": max(records, key=lambda x: x["image_mean_miou_exclude"]),
              "best_global": max(records, key=lambda x: x["global_miou"]),
              "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_global"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--model207", required=True)
    parser.add_argument("--model209", required=True)
    parser.add_argument("--model211", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
