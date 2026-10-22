#!/usr/bin/env python3
"""Re-average every saved 20-image validation score vector under both empty rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def walk(value, source: Path, trail: str = ""):
    if isinstance(value, dict):
        zero = value.get("image_scores")
        skipped = value.get("image_scores_exclude")
        if isinstance(zero, list) and isinstance(skipped, list) and len(zero) == len(skipped) == 20:
            row = {key: item for key, item in value.items()
                   if not isinstance(item, (list, dict))}
            row.update({
                "source": str(source),
                "json_path": trail or "$",
                "recomputed_empty_zero": float(np.mean(np.asarray(zero, dtype=np.float64))),
                "recomputed_empty_skipped": float(np.mean(np.asarray(skipped, dtype=np.float64))),
                "image_count": 20,
            })
            yield row
        for key, child in value.items():
            yield from walk(child, source, f"{trail}.{key}" if trail else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, source, f"{trail}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows, errors = [], []
    files = sorted(args.outputs.rglob("results.json"))
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(walk(obj, path))
        except Exception as exc:
            errors.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})

    zero_rank = sorted(rows, key=lambda row: row["recomputed_empty_zero"], reverse=True)
    skip_rank = sorted(rows, key=lambda row: row["recomputed_empty_skipped"], reverse=True)
    inconsistent = []
    for row in rows:
        for field, key in (("image_mean_miou", "recomputed_empty_zero"),
                           ("image_mean_miou_exclude", "recomputed_empty_skipped")):
            if field in row and abs(float(row[field]) - row[key]) > 1e-9:
                inconsistent.append({"source": row["source"], "json_path": row["json_path"],
                                    "field": field, "reported": row[field],
                                    "recomputed": row[key]})
    result = {
        "metric": "arithmetic mean of saved per-image scores; 20 validation images",
        "results_json_files_scanned": len(files),
        "20_image_score_vectors_recomputed": len(rows),
        "top_empty_zero": zero_rank[:30],
        "top_empty_skipped": skip_rank[:30],
        "inconsistencies_over_1e-9": inconsistent,
        "parse_errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "results_json_files_scanned", "20_image_score_vectors_recomputed",
        "top_empty_zero", "top_empty_skipped", "inconsistencies_over_1e-9",
        "parse_errors")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
