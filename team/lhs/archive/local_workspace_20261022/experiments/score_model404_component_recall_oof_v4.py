#!/usr/bin/env python3
"""Score v4 paired OOF and guard tiny Al3Ni/Eutectic components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import score_model404_component_recall_oof as shared  # noqa: E402


TARGET_CLASSES = {1: "Al3Ni", 3: "Eutectic Si"}
AREA_BINS = {"tiny_1_to_lt64": (1, 64), "medium_64_to_lt256": (64, 256)}
COMPONENT_BINS = {
    f"{class_name}_{bin_name}": limits
    for class_name in TARGET_CLASSES.values()
    for bin_name, limits in AREA_BINS.items()
}
base_hit_rate_delta_bootstrap = shared.hit_rate_delta_bootstrap


def component_hits(probability: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, int]]:
    prediction = np.asarray(probability).argmax(axis=0)
    result = {name: {"components": 0, "hits": 0} for name in COMPONENT_BINS}
    for class_id, class_name in TARGET_CLASSES.items():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(truth == class_id, dtype=np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            bin_name = next(
                (name for name, (lo, hi) in AREA_BINS.items() if lo <= area < hi), None
            )
            if bin_name is None:
                continue
            key = f"{class_name}_{bin_name}"
            component = labels == component_id
            result[key]["components"] += 1
            result[key]["hits"] += int(float(np.mean(prediction[component] == class_id)) >= 0.50)
    return result


def hit_rate_delta_bootstrap(candidate: dict, control: dict, samples: list[np.ndarray],
                             bin_name: str) -> dict:
    result = base_hit_rate_delta_bootstrap(candidate, control, samples, bin_name)
    class_name = next(name for name in TARGET_CLASSES.values if bin_name.startswith(name + "_"))
    size_name = bin_name[len(class_name) + 1:]
    # The original metric helper is unchanged; replace its single-class wording
    # with the actual target class and size interval in this versioned scorer.
    interval = AREA_BINS[size_name]
    result["hit_definition"] = (
        f"at least 50% of native-grid GT component pixels predicted as {class_name}; "
        f"area [{interval[0]},{interval[1]})"
    )
    return result


def score(*args, **kwargs) -> dict:
    shared.COMPONENT_BINS = COMPONENT_BINS
    shared.component_hits = component_hits
    shared.hit_rate_delta_bootstrap = hit_rate_delta_bootstrap
    result = shared.score(*args, **kwargs)
    checks = result["preregistered_gate"]["checks"]
    class_delta = result["paired_candidate_vs_control"]["class_delta_pp"]
    targeted_class_nonnegative = all(class_delta[name] >= 0 for name in TARGET_CLASSES.values())
    checks["targeted_class_iou_delta_nonnegative"] = targeted_class_nonnegative
    checks["all_class_delta_ge_minus_0_50pp_and_targeted_classes_nonnegative"] = (
        min(class_delta.values()) >= -0.50 and targeted_class_nonnegative
    )
    # The old single-class gate is retained in the immutable base scorer but
    # superseded here by the two-target-class gate above.
    checks.pop("all_class_delta_ge_minus_0_50pp_and_eutectic_nonnegative", None)
    result["native_targeted_component_hit_rates"] = result.pop(
        "native_eutectic_component_hit_rates"
    )
    result["targeted_component_classes"] = list(TARGET_CLASSES.values())
    result["experiment"] = "Model404 v4 Al3Ni + Eutectic-Si tiny-component recall fine-tuning"
    result["preregistered_gate"]["pass"] = all(checks.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite prior Model404 v4 score: {args.out}")
    result = score(args.data_root, args.candidate_root, args.control_root, args.baseline_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "candidate_oof_percent": result["candidate_scores"]["image_mean_miou_percent"],
        "matched_control_oof_percent": result["control_scores"]["image_mean_miou_percent"],
        "delta_pp": result["paired_candidate_vs_control"]["delta_image_mean_pp"],
        "group_bootstrap_95ci_pp": result["paired_candidate_vs_control"]["paired_group_bootstrap_95ci_pp"],
        "targeted_component_hit_rates": result["native_targeted_component_hit_rates"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
