#!/usr/bin/env python3
"""Paired 70-image, four-fold OOF comparison for the pre-fixed B4 soup."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    outputs = args.root / "outputs"
    sources = {
        "326": outputs / "model326_hybrid_fullstats_group_oof",
        "329": outputs / "model329_fullstats_width4000_uniform_group_oof",
        "soup": outputs / "model349_b4_soup_probe",
    }
    rows = []
    for fold in range(1, 5):
        fold_scores = {}
        for name, base in sources.items():
            folder = (base / f"alpha050_fold{fold}_oof" if name == "soup" else
                      base / f"fold{fold}_oof")
            scores = json.loads((folder / "scores.json").read_text(encoding="utf-8"))
            fold_scores[name] = scores["images"]
        names = set(fold_scores["326"])
        if names != set(fold_scores["329"]) or names != set(fold_scores["soup"]):
            raise ValueError(f"fold{fold}: paired image sets differ")
        for image in sorted(names):
            rows.append({"image": image, "group": image.split("_step", 1)[0],
                         "fold": fold, **{name: float(values[image]["miou"])
                                         for name, values in fold_scores.items()}})
    if len(rows) != 70 or len({row["image"] for row in rows}) != 70:
        raise ValueError("expected 70 distinct OOF images")
    fold_summary = []
    for fold in range(1, 5):
        subset = [row for row in rows if row["fold"] == fold]
        means = {name: float(np.mean([row[name] for row in subset])) for name in sources}
        fold_summary.append({"fold": fold, "images": len(subset), "means": means,
                             "soup_minus_329_pp": means["soup"] - means["329"]})
    means = {name: float(np.mean([row[name] for row in rows])) for name in sources}
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    groups = list(sorted(by_group))
    rng = np.random.default_rng(349)
    deltas = np.empty(10000, dtype=float)
    for idx in range(len(deltas)):
        draw = rng.choice(groups, len(groups), replace=True)
        chosen = [row for group in draw for row in by_group[group]]
        deltas[idx] = np.mean([row["soup"] - row["329"] for row in chosen])
    gate = (means["soup"] >= means["329"] and
            sum(item["soup_minus_329_pp"] > 0 for item in fold_summary) >= 3 and
            all(item["soup_minus_329_pp"] >= -0.5 for item in fold_summary))
    result = {
        "metric": "official image-wise four-class mIoU, empty-union=0, percent",
        "images": len(rows), "specimen_groups": len(groups),
        "mean": means, "folds": fold_summary,
        "soup_minus_329_pp": means["soup"] - means["329"],
        "group_bootstrap_95_ci_pp": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "preregistered_oof_gate_pass": gate,
        "per_image": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "images", "specimen_groups", "mean", "folds", "soup_minus_329_pp",
        "group_bootstrap_95_ci_pp", "preregistered_oof_gate_pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
