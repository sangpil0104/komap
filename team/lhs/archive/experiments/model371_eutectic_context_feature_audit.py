#!/usr/bin/env python3
"""Train-OOF diagnostic of Al3Ni context for Eutectic false positives.

This tests whether local predicted Al3Ni probability contains group-held-out
ranking information about remote Eutectic false positives beyond the raw
Eutectic-vs-Al3Ni margin. It reads only grouped train-OOF probabilities and
train masks; it does not fit a correction or inspect valid/test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

import model368_model329_oof_class_boundary_geometry as model368

SOURCES = {
    "322": "outputs/model322_team_b4_group_oof/train_oof_probs.npz",
    "329": "outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz",
    "333_control": "outputs/model333_b4_imagewise_lovasz_group_oof/control/train_oof_probs.npz",
    "333_lovasz": "outputs/model333_b4_imagewise_lovasz_group_oof/lovasz/train_oof_probs.npz",
}
RADII = (7, 15, 31)
STRUCTURE_8 = np.ones((3, 3), dtype=bool)
CLASS_ORDER = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")
EPS = 1e-7


def binary_auc(positive_scores: np.ndarray,
               negative_scores: np.ndarray) -> float | None:
    if not len(positive_scores) or not len(negative_scores):
        return None
    scores = np.concatenate((positive_scores, negative_scores))
    ranks = rankdata(scores, method="average")
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    edge = mask & ~ndimage.binary_erosion(mask, structure=STRUCTURE_8,
                                          border_value=0)
    return ndimage.binary_dilation(edge, structure=STRUCTURE_8,
                                   iterations=radius)


def pixel_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    p = np.maximum(probabilities.astype(np.float32, copy=False), EPS)
    logp = np.log(p)
    eut_vs_al3ni = logp[3] - logp[1]
    eut_vs_any = logp[3] - np.max(logp[:3], axis=0)
    features = {
        "low_eut_vs_al3ni_margin": -eut_vs_al3ni,
        "low_eut_vs_any_other_margin": -eut_vs_any,
    }
    for radius in RADII:
        width = 2 * radius + 1
        window = float(width * width)
        mean = ndimage.uniform_filter(p[1], size=width, mode="reflect")
        # Exclude the candidate pixel itself: this is its surrounding Al3Ni
        # context, not simply the pointwise runner-up probability again.
        features[f"neighbor_al3ni_probability_r{radius}"] = (
            window * mean - p[1]) / (window - 1.0)
    return features


def bootstrap_difference(differences: np.ndarray, seed: int,
                         repetitions: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(differences), size=len(differences))
        draws[index] = differences[sample].mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize_group_aucs(group_values: dict[str, dict[str, dict[str, float | None]]],
                         bootstrap: int, seed: int) -> dict:
    metrics = sorted({metric for row in group_values.values() for metric in row})
    summary = {}
    for metric in metrics:
        features = sorted({feature for row in group_values.values()
                           for feature in row.get(metric, {})})
        summary[metric] = {}
        for feature in features:
            available = [(group, row[metric][feature])
                         for group, row in group_values.items()
                         if row.get(metric, {}).get(feature) is not None]
            values = np.asarray([value for _, value in available], dtype=np.float64)
            summary[metric][feature] = {
                "groups_with_auc": int(len(values)),
                "group_macro_auc": float(values.mean()) if len(values) else None,
                "group_bootstrap_95ci": bootstrap_difference(
                    values, seed + len(summary[metric]), bootstrap)
                    if len(values) >= 2 else None,
            }

        # Paired comparison is the primary check: does surrounding Al3Ni
        # context rank remote FP above TP better than the pointwise pair margin?
        if metric == "far_eutectic_fp_vs_tp":
            baseline = "low_eut_vs_al3ni_margin"
            for feature in features:
                if feature == baseline:
                    continue
                paired = [(group, row[metric].get(feature),
                           row[metric].get(baseline))
                          for group, row in group_values.items()]
                paired = [(group, value, base) for group, value, base in paired
                          if value is not None and base is not None]
                delta = np.asarray([value - base for _, value, base in paired],
                                   dtype=np.float64)
                summary[metric][feature]["paired_delta_vs_pair_margin"] = (
                    float(delta.mean()) if len(delta) else None)
                summary[metric][feature]["paired_delta_group_bootstrap_95ci"] = (
                    bootstrap_difference(delta, seed + 100 + len(summary[metric]),
                                         bootstrap)
                    if len(delta) >= 2 else None)
                summary[metric][feature]["paired_groups"] = int(len(delta))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--expected-groups", type=int, default=59)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be >=100")

    data_root = args.data or args.root / "data"
    archives = {}
    for name, relative in SOURCES.items():
        path = args.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        archives[name] = np.load(path, allow_pickle=False)

    try:
        key_sets = {name: set(archive.files) for name, archive in archives.items()}
        first_name = next(iter(archives))
        names = sorted(key_sets[first_name])
        for source, keys in key_sets.items():
            if keys != key_sets[first_name]:
                raise ValueError(f"OOF image-key mismatch for source {source}")
        if len(names) != args.expected_images:
            raise ValueError(f"expected {args.expected_images} images, found {len(names)}")
        groups = {name: model368.group_for(name) for name in names}
        unique_groups = sorted(set(groups.values()))
        if len(unique_groups) != args.expected_groups:
            raise ValueError(f"expected {args.expected_groups} groups, found {len(unique_groups)}")

        # Store only candidate-pixel scores, grouped by specimen, so group-level
        # AUC and bootstrap never treat pixels as independent observations.
        pooled: dict[str, dict[str, dict[str, dict[str, list[np.ndarray]]]]] = {
            source: {group: {} for group in unique_groups} for source in archives
        }
        totals = {source: {"pred_eut_pixels": 0, "true_eut_pixels": 0,
                           "eut_fp_pixels": 0, "far_eut_fp_pixels": 0,
                           "al3ni_source_fp_pixels": 0}
                  for source in archives}
        masks = data_root / "train" / "masks"
        for image_index, image_name in enumerate(names):
            target = model368.load_target(masks / f"{image_name}_mask.png")
            eut_gt = target == 3
            near_band = boundary_band(eut_gt, radius=2)
            group = groups[image_name]
            for source, archive in archives.items():
                probabilities = np.asarray(archive[image_name], dtype=np.float32)
                if probabilities.ndim != 3 or probabilities.shape[0] != 4:
                    raise ValueError(f"expected 4xHxW maps for {source}/{image_name}")
                if probabilities.shape[1:] != target.shape:
                    raise ValueError(f"mask/probability shape mismatch for {source}/{image_name}")
                if not np.allclose(probabilities.sum(0), 1.0, atol=3e-3):
                    raise ValueError(f"invalid probabilities for {source}/{image_name}")
                prediction = probabilities.argmax(axis=0)
                candidate = prediction == 3
                true_positive = candidate & eut_gt
                false_positive = candidate & ~eut_gt
                far_fp = false_positive & ~near_band
                al3ni_fp = candidate & (target == 1)
                features = pixel_scores(probabilities)
                candidate_features = {
                    key: value[candidate].astype(np.float32, copy=False)
                    for key, value in features.items()
                }
                local = pooled[source][group]
                for metric_name, positive in (
                    ("all_eutectic_fp_vs_tp", false_positive),
                    ("far_eutectic_fp_vs_tp", far_fp),
                    ("al3ni_source_fp_vs_tp", al3ni_fp),
                ):
                    selected_positive = positive[candidate]
                    selected_tp = true_positive[candidate]
                    metric_row = local.setdefault(metric_name, {})
                    for feature_name, scores in candidate_features.items():
                        values = metric_row.setdefault(feature_name,
                                                       {"positive": [], "negative": []})
                        values["positive"].append(scores[selected_positive])
                        values["negative"].append(scores[selected_tp])
                total = totals[source]
                total["pred_eut_pixels"] += int(candidate.sum())
                total["true_eut_pixels"] += int(true_positive.sum())
                total["eut_fp_pixels"] += int(false_positive.sum())
                total["far_eut_fp_pixels"] += int(far_fp.sum())
                total["al3ni_source_fp_pixels"] += int(al3ni_fp.sum())
            if (image_index + 1) % 10 == 0 or image_index + 1 == len(names):
                print(f"audited context features {image_index + 1}/{len(names)}",
                      flush=True)

        group_aucs = {}
        for source, per_group in pooled.items():
            group_aucs[source] = {}
            for group, metrics in per_group.items():
                group_aucs[source][group] = {}
                for metric_name, feature_rows in metrics.items():
                    group_aucs[source][group][metric_name] = {}
                    for feature_name, values in feature_rows.items():
                        positive = np.concatenate(values["positive"])
                        negative = np.concatenate(values["negative"])
                        group_aucs[source][group][metric_name][feature_name] = binary_auc(
                            positive, negative)

        summary = {
            source: summarize_group_aucs(rows, args.bootstrap, seed=371 + index * 1000)
            for index, (source, rows) in enumerate(group_aucs.items())
        }
        # Conservative pre-registered screening rule, not a score claim:
        # both Model329 and Model322 must show >=.03 AUC gain with a paired
        # specimen-group bootstrap lower bound above zero on remote FP vs TP.
        candidates = ("neighbor_al3ni_probability_r15",)
        primary_deltas = {}
        for source in ("329", "322"):
            primary_deltas[source] = summary[source]["far_eutectic_fp_vs_tp"][
                candidates[0]]
        passes = all(
            primary_deltas[source]["paired_groups"] >= 30
            and primary_deltas[source]["paired_delta_vs_pair_margin"] is not None
            and primary_deltas[source]["paired_delta_vs_pair_margin"] >= 0.03
            and primary_deltas[source]["paired_delta_group_bootstrap_95ci"][0] > 0.0
            for source in ("329", "322")
        )
        result = {
            "model": 371,
            "audit": "train-OOF Al3Ni neighborhood feature ranking for Eutectic false positives; no correction trained",
            "metric_scope": "group-macro AUC over held-out train OOF predictions; bootstrap resamples 59 specimen groups, not pixels",
            "class_order": list(CLASS_ORDER),
            "inputs": {source: str(args.root / relative) for source, relative in SOURCES.items()},
            "input_scope": "grouped train-OOF probabilities plus data/train/masks only; no valid/test files read",
            "images": len(names),
            "specimen_groups": len(unique_groups),
            "features": {
                "low_eut_vs_al3ni_margin": "negative log(p_Eut)-log(p_Al3Ni); larger should indicate Al3Ni confusion",
                "low_eut_vs_any_other_margin": "negative log(p_Eut)-max(log(p_Al),log(p_Al3Ni),log(p_Primary)); larger should indicate any Eutectic FP",
                "neighbor_al3ni_probability_r7_r15_r31": "mean p_Al3Ni in a square neighborhood of radius r, with candidate-center pixel excluded",
            },
            "far_fp_definition": "predicted Eutectic false positives outside the 2px 8-connected dilation of the GT Eutectic inner boundary",
            "totals": totals,
            "group_macro_auc": summary,
            "primary_screen": {
                "feature": candidates[0],
                "comparison": "far Eutectic FP-vs-TP group-macro AUC gain over pointwise Eutectic-vs-Al3Ni margin",
                "required_sources": ["329", "322"],
                "pass_if_each_source_gain_at_least": 0.03,
                "and_paired_59_group_bootstrap_lower_ci_above": 0.0,
                "source_results": primary_deltas,
                "continue_to_nested_group_cv": bool(passes),
            },
            "caveat": "The OOF sources share training data and are not independent replications. AUC is diagnostic only; it does not establish mIoU gain or transfer to Model360. Any correction requires nested group-held-out threshold selection and a route-source transfer check before one final valid evaluation.",
        }
    finally:
        for archive in archives.values():
            archive.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"model": 371, "totals": totals,
                      "primary_screen": result["primary_screen"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
