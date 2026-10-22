#!/usr/bin/env python3
"""Per-image high-source ablation from two complete valid inference runs."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path


MAIN = "restart51_aspp_768_finetune"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--forced", type=Path, required=True)
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    forced = json.loads(args.forced.read_text())
    chosen = dict(re.findall(r"DONE base (\S+) source=(\S+)", args.base_log.read_text()))
    old = {row["image"]: row["image_miou"] for row in baseline["images"]}
    new = {row["image"]: row["image_miou"] for row in forced["images"]}
    if set(old) != set(new) or set(old) != set(chosen) or len(old) != 20:
        raise ValueError("Baseline, forced, and selected-source images differ")
    others = sorted(set(chosen.values()) - {MAIN})
    groups = {}
    for source in others:
        names = sorted(name for name in old if chosen[name] == source)
        changes = {name: 100.0 * (new[name] - old[name]) for name in names}
        groups[source] = {"images": names, "delta_per_image_percentage_points": changes,
                          "mean_score_delta_percentage_points": sum(changes.values()) / 20,
                          "worst_image_delta_percentage_points": min(changes.values())}
    combinations = []
    for size in range(len(others) + 1):
        for dropped in itertools.combinations(others, size):
            score = sum(new[name] if chosen[name] in dropped else old[name] for name in old) / len(old)
            affected = [name for name in old if chosen[name] in dropped]
            worst = min((100.0 * (new[name] - old[name]) for name in affected), default=0.0)
            combinations.append({"drop_high_sources": list(dropped), "checkpoint_count": 15 - size,
                                 "affected_images": len(affected), "image_miou": score,
                                 "delta_vs_baseline_percentage_points": 100.0 * (score - baseline["image_miou"]),
                                 "worst_affected_image_delta_percentage_points": worst})
    output = {"metric": "valid20 image-wise 4-class mIoU, empty union=0",
              "evaluation_role": "exploratory post-hoc hybrid of two complete image-independent runs",
              "baseline": baseline["image_miou"], "forced_all": forced["image_miou"],
              "selected_sources": chosen, "groups": groups,
              "combinations": sorted(combinations, key=lambda x: (-x["image_miou"], x["checkpoint_count"]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"groups": groups, "top_combinations": output["combinations"][:8]}, indent=2))


if __name__ == "__main__":
    main()
