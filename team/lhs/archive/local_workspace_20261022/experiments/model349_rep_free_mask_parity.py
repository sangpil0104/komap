#!/usr/bin/env python3
"""Compare fixed Model349 masks from historical and rep-free raw-image caches."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import model349_source_prune_audit as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--compact-entries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    outputs = args.root / "outputs"
    result = lambda name: json.loads((outputs / name / "results.json").read_text())
    route = result("model339_joint_gamma_route_search")["best_image_empty_zero"]
    gamma329 = tuple(result("model341_model329_joint_route_search")["best_image_empty_zero"]["gamma329"])
    gamma336 = tuple(result("model346_model336_into_341_route")["raw_best_gamma336"])
    gamma347 = tuple(result("model349_model347_into_346_route")["raw_best_gamma347"])
    variant = next(audit.setting_variants(route, gamma329, gamma336, gamma347))
    rows = audit.load_rows(args.root)
    with args.compact_entries.open("rb") as stream:
        compact = pickle.load(stream)
    if len(rows) != len(compact):
        raise ValueError("entry count differs")
    records = []
    for row, item in zip(rows, compact):
        if not np.array_equal(row["item"]["target"], item["target"]):
            raise ValueError(f"target/order mismatch for {row['name']}")
        guide = audit.model341.route.bilateral_weights(row["item"]["z"])
        historical_mask = audit.prediction(row, variant, guide)
        direct_row = {"item": item, "base": item["preaux"], "sources": row["sources"]}
        rep_free_mask = audit.prediction(direct_row, variant, guide)
        different = int(np.count_nonzero(historical_mask != rep_free_mask))
        records.append({"image": row["name"], "pixels": int(historical_mask.size),
                        "different_pixels": different,
                        "identical": different == 0})
        print(f"{row['name']}: {different} different pixels", flush=True)
    result_json = {
        "images": len(records),
        "identical_images": sum(item["identical"] for item in records),
        "different_pixels": sum(item["different_pixels"] for item in records),
        "total_pixels": sum(item["pixels"] for item in records),
        "representation_forward_calls_for_compact_base": 0,
        "per_image": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_json, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result_json[key] for key in (
        "images", "identical_images", "different_pixels", "total_pixels")}), flush=True)


if __name__ == "__main__":
    main()
