#!/usr/bin/env python3
"""Paired Model404 OOF score with native-grid component-hit safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model397_reproduction_bundle"
sys.path.insert(0, str(BUNDLE / "scripts"))
from score_model397_oof import (  # noqa: E402
    CLASS_NAMES,
    FOLD_FILE,
    bootstrap_indices,
    class_ious,
    compare,
    group_and_step,
    summarize,
)
from write_oof_manifest import sha256  # noqa: E402


PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)], dtype=np.uint8
)
COMPONENT_BINS = {"tiny_1_to_lt64": (1, 64), "small_64_to_lt256": (64, 256)}


def read_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    truth = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        truth[np.all(rgb == color, axis=2)] = class_id
    if np.any(truth == 255):
        raise ValueError(f"unknown RGB class color in {path}")
    return truth


def load_archive(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(f"OOF inventory mismatch at {path}: expected {len(expected)}, got {len(archive.files)}")
        return {key: np.asarray(archive[key], dtype=np.float32) for key in archive.files}


def component_hits(probability: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, int]]:
    prediction = np.asarray(probability).argmax(axis=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(truth == 3, dtype=np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    result = {name: {"components": 0, "hits": 0} for name in COMPONENT_BINS}
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        name = next((key for key, (lo, hi) in COMPONENT_BINS.items() if lo <= area < hi), None)
        if name is None:
            continue
        component = labels == component_id
        result[name]["components"] += 1
        result[name]["hits"] += int(float(np.mean(prediction[component] == 3)) >= 0.50)
    return result


def hit_rate_delta_bootstrap(candidate: dict, control: dict, samples: list[np.ndarray], bin_name: str) -> dict:
    candidate_components = np.asarray([row[bin_name]["components"] for row in candidate], dtype=np.float64)
    control_components = np.asarray([row[bin_name]["components"] for row in control], dtype=np.float64)
    candidate_hits = np.asarray([row[bin_name]["hits"] for row in candidate], dtype=np.float64)
    control_hits = np.asarray([row[bin_name]["hits"] for row in control], dtype=np.float64)
    n_c = int(candidate_components.sum())
    n_b = int(control_components.sum())
    if n_c != n_b or n_c == 0:
        raise ValueError(f"component support mismatch/empty for {bin_name}: candidate={n_c}, control={n_b}")
    candidate_rate = float(candidate_hits.sum() / n_c)
    control_rate = float(control_hits.sum() / n_b)
    boot = []
    for sample in samples:
        denom_c = candidate_components[sample].sum()
        denom_b = control_components[sample].sum()
        if denom_c == 0 or denom_b == 0:
            continue
        boot.append(
            candidate_hits[sample].sum() / denom_c - control_hits[sample].sum() / denom_b
        )
    if len(boot) < 100:
        raise ValueError(f"too few nonempty group-bootstrap samples for {bin_name}")
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
    return {
        "components": n_c,
        "candidate_hit_rate_percent": candidate_rate * 100.0,
        "control_hit_rate_percent": control_rate * 100.0,
        "delta_pp": (candidate_rate - control_rate) * 100.0,
        "paired_group_bootstrap_95ci_pp": [float(lo * 100.0), float(hi * 100.0)],
        "hit_definition": "at least 50% of GT component pixels predicted as Eutectic Si",
        "connectivity": 8,
    }


def score(data_root: Path, candidate_root: Path, control_root: Path,
          baseline_root: Path) -> dict:
    data_root = data_root.resolve()
    candidate_root = candidate_root.resolve()
    control_root = control_root.resolve()
    baseline_root = baseline_root.resolve()
    image_paths = sorted((data_root / "train/images").glob("*_image.png"))
    if len(image_paths) != 70:
        raise ValueError(f"expected exactly 70 train images, found {len(image_paths)}")
    ids = [path.name.removesuffix("_image.png") for path in image_paths]
    info = {image_id: group_and_step(image_id) for image_id in ids}
    if len({group for group, _ in info.values()}) != 59:
        raise ValueError("expected exactly 59 train specimen groups")
    fold_cfg = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    fold_for_group: dict[str, int] = {}
    for entry in fold_cfg["folds"]:
        for group in entry["heldout_groups"]:
            fold = int(entry["fold"])
            if group in fold_for_group:
                raise ValueError(f"group assigned to multiple folds: {group}")
            fold_for_group[group] = fold
    if set(fold_for_group) != {group for group, _ in info.values()}:
        raise ValueError("fixed fold manifest does not cover current training groups")

    rows: list[dict] = []
    component_rows = {arm: [] for arm in ("candidate", "control")}
    for fold in range(1, 5):
        expected = {image_id for image_id in ids if fold_for_group[info[image_id][0]] == fold}
        archives = {
            "candidate": load_archive(candidate_root / f"fold{fold}_oof/valid_probabilities.npz", expected),
            "control": load_archive(control_root / f"fold{fold}_oof/valid_probabilities.npz", expected),
            "model392": load_archive(
                baseline_root / f"seed1/fold{fold}_oof/valid_probabilities.npz", expected
            ),
        }
        for image_id in sorted(expected):
            truth = read_mask(data_root / "train/masks" / f"{image_id}_mask.png")
            group, step = info[image_id]
            item = {"fold": fold, "image_id": image_id, "group": group, "step": step}
            for arm, probability in archives.items():
                item[arm] = class_ious(probability[image_id], truth)
            rows.append(item)
            for arm in component_rows:
                component_rows[arm].append(component_hits(archives[arm][image_id], truth))

    if len(rows) != 70 or len({row["image_id"] for row in rows}) != 70:
        raise ValueError("expected each train image to be scored exactly once out-of-fold")
    samples = bootstrap_indices(rows, draws=20_000, seed=404)
    score_arrays = {
        arm: np.asarray([row[arm] for row in rows], dtype=np.float64)
        for arm in ("candidate", "control", "model392")
    }
    treatment_control = compare(score_arrays["candidate"], score_arrays["control"], rows, samples)
    treatment_parent = compare(score_arrays["candidate"], score_arrays["model392"], rows, samples)
    control_parent = compare(score_arrays["control"], score_arrays["model392"], rows, samples)
    component_metrics = {
        name: hit_rate_delta_bootstrap(
            component_rows["candidate"], component_rows["control"], samples, name
        )
        for name in COMPONENT_BINS
    }
    gate = {
        "candidate_vs_matched_control_gain_ge_0_20pp": treatment_control["delta_image_mean_pp"] >= 0.20,
        "candidate_vs_control_group_ci_lower_gt_zero": treatment_control["paired_group_bootstrap_95ci_pp"][0] > 0,
        "candidate_vs_control_positive_at_least_3_folds": treatment_control["positive_folds"] >= 3,
        "candidate_vs_control_worst_fold_ge_minus_0_30pp": min(treatment_control["fold_delta_pp"].values()) >= -0.30,
        "all_class_delta_ge_minus_0_50pp_and_eutectic_nonnegative": (
            min(treatment_control["class_delta_pp"].values()) >= -0.50
            and treatment_control["class_delta_pp"]["Eutectic Si"] >= 0
        ),
        "step4_delta_ge_minus_0_30pp": (
            treatment_control["step4_delta_pp"] is not None
            and treatment_control["step4_delta_pp"] >= -0.30
        ),
        "tiny_and_small_medium_component_hit_rates_safe": all(
            metric["delta_pp"] >= 0
            and metric["paired_group_bootstrap_95ci_pp"][0] >= -5.0
            for metric in component_metrics.values()
        ),
    }
    return {
        "experiment": "Model404 Eutectic-Si component-normalized recall fine-tuning",
        "metric": "image-wise mean of four class IoUs; empty-union=0",
        "class_order": CLASS_NAMES,
        "images": 70,
        "specimen_groups": 59,
        "folds": 4,
        "primary_comparison": "component treatment vs compute-matched continuation control",
        "candidate_scores": summarize(score_arrays["candidate"]),
        "control_scores": summarize(score_arrays["control"]),
        "model392_parent_scores": summarize(score_arrays["model392"]),
        "paired_candidate_vs_control": treatment_control,
        "paired_candidate_vs_model392_parent": treatment_parent,
        "paired_control_vs_model392_parent": control_parent,
        "native_eutectic_component_hit_rates": component_metrics,
        "preregistered_gate": {"checks": gate, "pass": all(gate.values())},
        "valid_read": False,
        "test_read": False,
        "fold_manifest_sha256": sha256(FOLD_FILE),
        "interpretation": "Selection-conditional train-OOF development evidence; not official validation/test performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite prior Model404 score: {args.out}")
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
        "component_hit_rates": result["native_eutectic_component_hit_rates"],
        "gate_pass": result["preregistered_gate"]["pass"],
        "valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
