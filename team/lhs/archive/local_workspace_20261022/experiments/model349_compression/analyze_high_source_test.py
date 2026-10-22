#!/usr/bin/env python3
"""Compare test masks for brightness-source pruning without using test labels."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


MAIN = "restart51_aspp_768_finetune"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-labels", type=Path, required=True)
    parser.add_argument("--forced-labels", type=Path, required=True)
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen = dict(re.findall(r"DONE base (\S+) source=(\S+)", args.base_log.read_text()))
    old_files = {path.stem: path for path in args.baseline_labels.glob("*.png")}
    new_files = {path.stem: path for path in args.forced_labels.glob("*.png")}
    if set(old_files) != set(new_files) or set(old_files) != set(chosen) or len(old_files) != 10:
        raise ValueError("Expected matching baseline, forced, and source sets of ten images")
    images = {}
    for name in sorted(chosen):
        baseline = np.array(Image.open(old_files[name]))
        forced = np.array(Image.open(new_files[name]))
        if baseline.shape != forced.shape:
            raise ValueError(name)
        images[name] = {"source": chosen[name], "pixels": int(baseline.size),
                        "different_pixels": int(np.count_nonzero(baseline != forced))}
    others = sorted(set(chosen.values()) - {MAIN})
    combinations = []
    total = sum(row["pixels"] for row in images.values())
    for size in range(len(others) + 1):
        for dropped in itertools.combinations(others, size):
            changed = sum(row["different_pixels"] for row in images.values() if row["source"] in dropped)
            combinations.append({"drop_high_sources": list(dropped), "checkpoint_count": 15 - size,
                                 "different_pixels": changed, "total_pixels": total,
                                 "agreement": 1.0 - changed / total})
    result = {"comparison": "baseline versus forced-restart51 test masks; no test ground truth",
              "images": images, "combinations": combinations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
