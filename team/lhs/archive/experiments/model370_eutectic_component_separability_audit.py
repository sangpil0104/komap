#!/usr/bin/env python3
"""Train-OOF-only diagnostic: can low-confidence small Eutectic components be removed?

This is a separability audit, not a validation score. It reads Model329's 70
grouped train-OOF probability maps and train masks only. No valid/test files are
opened. Component-level confidence thresholds are normalized within each image
to avoid relying on an absolute probability scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

import model368_model329_oof_class_boundary_geometry as model368

CLASS_ORDER = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")
COMPONENT_SIZES = (16, 32, 64, 128, 256)
MARGIN_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
EXPECTED_MODEL367_OOF_MIOU = 0.7912326741291695
COMPONENT_SIZE_BINS = ((1, 16), (17, 32), (33, 64), (65, 128),
                       (129, 256), (257, None))
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.bincount(target.astype(np.int64).ravel() * 4
                       + prediction.astype(np.int64).ravel(),
                       minlength=16).reshape(4, 4)


def image_miou(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4, dtype=np.float64),
                    where=union > 0)
    return float(iou.mean()), iou


def component_margin_auc(samples: dict[str, dict[str, list]]) -> list[dict]:
    """AUC for low-margin scores separating mostly-FP from mostly-TP components."""
    results = []
    for label, values in samples.items():
        fp_scores = np.asarray(values["fp_scores"], dtype=np.float64)
        tp_scores = np.asarray(values["tp_scores"], dtype=np.float64)
        auc = None
        if len(fp_scores) and len(tp_scores):
            all_scores = np.concatenate((fp_scores, tp_scores))
            ranks = rankdata(all_scores, method="average")
            n_fp, n_tp = len(fp_scores), len(tp_scores)
            auc = float((ranks[:n_fp].sum() - n_fp * (n_fp + 1) / 2)
                        / (n_fp * n_tp))
        results.append({"component_size_bin_px": label,
                        "mostly_false_positive_components": int(len(fp_scores)),
                        "mostly_true_positive_components": int(len(tp_scores)),
                        "low_margin_fp_vs_tp_auc": auc})
    return results


def component_features(probabilities: np.ndarray, prediction: np.ndarray,
                       target: np.ndarray):
    labels, count = ndimage.label(prediction == 3, structure=STRUCTURE_8)
    sizes = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)
    logp = np.log(np.maximum(probabilities.astype(np.float32), 1e-7))
    alternative = logp.copy()
    alternative[3] = -np.inf
    second = alternative.argmax(axis=0).astype(np.uint8)
    margin = logp[3] - np.max(alternative, axis=0)
    eut_pixels = prediction == 3
    component_ids = np.arange(1, count + 1)
    counts = sizes[component_ids]
    means = np.bincount(labels[eut_pixels], weights=margin[eut_pixels],
                         minlength=count + 1)[component_ids] / np.maximum(counts, 1)
    if eut_pixels.any():
        sorted_pixel_margins = np.sort(margin[eut_pixels])
        mean_quantiles = np.searchsorted(sorted_pixel_margins, means, side="right") / len(
            sorted_pixel_margins)
    else:
        mean_quantiles = np.ones(count, dtype=np.float64)
    transition_codes = (labels[eut_pixels].astype(np.int64) * 16
                        + target[eut_pixels].astype(np.int64) * 4
                        + second[eut_pixels].astype(np.int64))
    transitions = np.bincount(transition_codes, minlength=(count + 1) * 16).reshape(
        count + 1, 4, 4)
    if not np.array_equal(transitions[1:].sum(axis=(1, 2)), counts):
        raise AssertionError("component transition counts do not match component areas")
    if not np.array_equal(transitions[1:].sum(axis=(0, 2)),
                          confusion(prediction, target)[:, 3]):
        raise AssertionError("component transitions do not match Eutectic confusion column")
    return component_ids, counts, mean_quantiles, transitions


def evaluate_rule(component_ids: np.ndarray, component_sizes: np.ndarray,
                  component_margin_quantiles: np.ndarray,
                  component_transitions: np.ndarray, max_area: int,
                  max_margin_quantile: float, base_matrix: np.ndarray,
                  all_eutectic_fp: int) -> dict:
    selected_ids = component_ids[
        (component_sizes <= max_area)
        & (component_margin_quantiles <= max_margin_quantile)]
    selected_counts = (component_transitions[selected_ids].sum(axis=0)
                       if len(selected_ids) else np.zeros((4, 4), dtype=np.int64))
    removed = int(selected_counts.sum())
    false_positive_by_gt = selected_counts[:3].sum(axis=1)
    fp_pixels = int(false_positive_by_gt.sum())
    tp_count = int(selected_counts[3].sum())
    second_correct = int(np.trace(selected_counts[:3, :3]))

    # Exact confusion update: every selected prediction moves from Eutectic
    # column 3 to its pixelwise runner-up class, without copying full masks.
    cleaned_matrix = base_matrix.copy()
    if selected_counts[:, 3].any():
        raise AssertionError("runner-up reassignment unexpectedly targets Eutectic")
    cleaned_matrix[:, 3] -= selected_counts.sum(axis=1)
    cleaned_matrix += selected_counts
    if np.any(cleaned_matrix < 0) or cleaned_matrix.sum() != base_matrix.sum():
        raise AssertionError("invalid cleaned confusion matrix")
    cleaned_score, cleaned_iou = image_miou(cleaned_matrix)

    # Optimistic ceiling: an oracle repairs only selected Eutectic false-positive
    # pixels, preserves selected true positives, and knows each pixel's GT class.
    oracle_matrix = base_matrix.copy()
    oracle_matrix[:3, 3] -= false_positive_by_gt
    oracle_matrix[np.arange(3), np.arange(3)] += false_positive_by_gt
    if np.any(oracle_matrix < 0) or oracle_matrix.sum() != base_matrix.sum():
        raise AssertionError("invalid oracle confusion matrix")
    oracle_score, _ = image_miou(oracle_matrix)
    return {
        "selected_components": int(len(selected_ids)),
        "selected_pixels": removed,
        "selected_fp_pixels": fp_pixels,
        "selected_tp_pixels": tp_count,
        "second_choice_correct_fp_pixels": second_correct,
        "selected_pixel_fp_purity": float(fp_pixels / removed) if removed else 0.0,
        "second_choice_correct_given_fp": float(second_correct / fp_pixels) if fp_pixels else 0.0,
        "fraction_of_all_eutectic_fp_covered": float(fp_pixels / all_eutectic_fp)
        if all_eutectic_fp else 0.0,
        "cleaned_image_miou_empty0": cleaned_score,
        "cleaned_class_iou_empty0": cleaned_iou.tolist(),
        "oracle_image_miou_empty0": oracle_score,
        "oracle_delta_pp": (oracle_score - image_miou(base_matrix)[0]) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=70)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    oof = args.oof or (args.root / "outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz")
    if not oof.is_file():
        raise FileNotFoundError(oof)

    with np.load(oof) as archive:
        names = sorted(archive.files)
        if len(names) != args.expected_images:
            raise ValueError(f"expected {args.expected_images} OOF maps, found {len(names)}")
        groups = [model368.group_for(name) for name in names]
        if len(set(groups)) != 59:
            raise ValueError(f"expected 59 specimen groups, got {len(set(groups))}")
        rule_records = [{"max_component_area_px": area,
                         "max_imagewise_margin_quantile": quantile,
                         "image_scores": [], "class_iou_sum": np.zeros(4),
                         "oracle_scores": [], "selected_fp_pixels": 0,
                         "selected_pixels": 0, "selected_tp_pixels": 0,
                         "all_eutectic_fp_pixels": 0,
                         "second_choice_correct_fp_pixels": 0,
                         "selected_component_count": 0}
                        for area in COMPONENT_SIZES for quantile in MARGIN_QUANTILES]
        baseline_scores = []
        baseline_class_iou = []
        per_image = []
        component_auc_samples = {
            f"{lower}-{upper if upper is not None else 'inf'}":
                {"fp_scores": [], "tp_scores": []}
            for lower, upper in COMPONENT_SIZE_BINS
        }
        masks_dir = args.root / "data/train/masks"
        for image_index, name in enumerate(names):
            probabilities = np.asarray(archive[name], dtype=np.float32)
            if probabilities.ndim != 3 or probabilities.shape[0] != 4:
                raise ValueError(f"expected 4xHxW map for {name}, got {probabilities.shape}")
            if not np.allclose(probabilities.sum(0), 1.0, atol=3e-3):
                raise ValueError(f"invalid probabilities for {name}")
            target = model368.load_target(masks_dir / f"{name}_mask.png")
            prediction = probabilities.argmax(0).astype(np.uint8)
            if prediction.shape != target.shape:
                raise ValueError(f"shape mismatch in {name}")
            base_matrix = confusion(prediction, target)
            base_score, base_iou = image_miou(base_matrix)
            baseline_scores.append(base_score)
            baseline_class_iou.append(base_iou)
            features = component_features(probabilities, prediction, target)
            ids, counts, quantiles, transitions = features
            all_fp = int(((prediction == 3) & (target != 3)).sum())
            component_sizes = counts
            component_fp_pixels = transitions[1:, :3, :].sum(axis=(1, 2))
            component_total_pixels = transitions[1:].sum(axis=(1, 2))
            component_fp_fraction = component_fp_pixels / np.maximum(
                component_total_pixels, 1)
            for component_index, area in enumerate(component_sizes):
                for lower, upper in COMPONENT_SIZE_BINS:
                    if area >= lower and (upper is None or area <= upper):
                        bin_name = f"{lower}-{upper if upper is not None else 'inf'}"
                        if component_fp_fraction[component_index] >= 0.80:
                            component_auc_samples[bin_name]["fp_scores"].append(
                                -float(quantiles[component_index]))
                        elif component_fp_fraction[component_index] <= 0.20:
                            component_auc_samples[bin_name]["tp_scores"].append(
                                -float(quantiles[component_index]))
                        break
            row_rules = []
            for index, record in enumerate(rule_records):
                area = record["max_component_area_px"]
                quantile = record["max_imagewise_margin_quantile"]
                values = evaluate_rule(ids, counts, quantiles, transitions,
                                       area, quantile, base_matrix, all_fp)
                record["image_scores"].append(values["cleaned_image_miou_empty0"])
                record["class_iou_sum"] += np.asarray(values["cleaned_class_iou_empty0"])
                record["oracle_scores"].append(values["oracle_image_miou_empty0"])
                record["selected_fp_pixels"] += values["selected_fp_pixels"]
                record["selected_pixels"] += values["selected_pixels"]
                record["selected_tp_pixels"] += values["selected_tp_pixels"]
                record["all_eutectic_fp_pixels"] += all_fp
                record["second_choice_correct_fp_pixels"] += values[
                    "second_choice_correct_fp_pixels"]
                record["selected_component_count"] += values["selected_components"]
                row_rules.append({
                    "max_component_area_px": area,
                    "max_imagewise_margin_quantile": quantile,
                    "selected_components": values["selected_components"],
                    "selected_pixels": values["selected_pixels"],
                    "selected_fp_pixels": values["selected_fp_pixels"],
                    "selected_tp_pixels": values["selected_tp_pixels"],
                    "selected_pixel_fp_purity": values["selected_pixel_fp_purity"],
                    "second_choice_correct_given_fp": values["second_choice_correct_given_fp"],
                    "fraction_of_all_eutectic_fp_covered": values[
                        "fraction_of_all_eutectic_fp_covered"],
                    "oracle_delta_pp": values["oracle_delta_pp"],
                })
            per_image.append({"image": name, "group": groups[image_index],
                              "baseline_image_miou_empty0": base_score,
                              "baseline_class_iou_empty0": base_iou.tolist(),
                              "predicted_eutectic_components": int(len(ids)),
                              "predicted_eutectic_pixels": int((prediction == 3).sum()),
                              "eutectic_fp_pixels": all_fp,
                              "component_rules": row_rules})
            if (image_index + 1) % 10 == 0 or image_index + 1 == len(names):
                print(f"audited Model329 OOF components {image_index + 1}/{len(names)}",
                      flush=True)

    baseline = float(np.mean(baseline_scores))
    if not np.isclose(baseline, EXPECTED_MODEL367_OOF_MIOU, rtol=0, atol=1e-10):
        raise AssertionError(
            f"Model329 OOF baseline mismatch: {baseline} vs {EXPECTED_MODEL367_OOF_MIOU}")
    records = []
    for record in rule_records:
        clean_scores = np.asarray(record.pop("image_scores"), dtype=np.float64)
        oracle_scores = np.asarray(record.pop("oracle_scores"), dtype=np.float64)
        class_iou_sum = record.pop("class_iou_sum")
        record["cleaned_image_mean_miou_empty0"] = float(clean_scores.mean())
        record["cleaned_delta_pp"] = float((clean_scores.mean() - baseline) * 100.0)
        record["cleaned_class_mean_iou_empty0"] = (class_iou_sum / len(names)).tolist()
        record["oracle_image_mean_miou_empty0"] = float(oracle_scores.mean())
        record["oracle_delta_pp"] = float((oracle_scores.mean() - baseline) * 100.0)
        record["selected_pixel_fp_purity"] = float(
            record["selected_fp_pixels"] / record["selected_pixels"]
            if record["selected_pixels"] else 0.0)
        record["second_choice_correct_given_fp"] = float(
            record["second_choice_correct_fp_pixels"] / record["selected_fp_pixels"]
            if record["selected_fp_pixels"] else 0.0)
        record["fraction_of_all_eutectic_fp_covered"] = float(
            record["selected_fp_pixels"] / record["all_eutectic_fp_pixels"]
            if record["all_eutectic_fp_pixels"] else 0.0)
        records.append(record)

    size_only_by_area = {
        row["max_component_area_px"]: row
        for row in records if row["max_imagewise_margin_quantile"] == 1.0
    }
    for row in records:
        size_only = size_only_by_area[row["max_component_area_px"]]
        row["same_area_size_only_delta_pp"] = size_only["cleaned_delta_pp"]
        row["confidence_gain_vs_same_area_size_only_pp"] = float(
            row["cleaned_delta_pp"] - size_only["cleaned_delta_pp"])

    best_oracle = max(records, key=lambda row: row["oracle_delta_pp"])
    eligible = [row for row in records
                if row["selected_pixel_fp_purity"] >= 0.80
                and row["second_choice_correct_given_fp"] >= 0.70
                and row["fraction_of_all_eutectic_fp_covered"] >= 0.10]
    eligible_with_oracle_headroom = [row for row in eligible
                                     if row["max_imagewise_margin_quantile"] < 1.0
                                     and row["oracle_delta_pp"] >= 0.15
                                     and row["cleaned_delta_pp"] > 0.0
                                     and row["confidence_gain_vs_same_area_size_only_pp"] > 0.0]
    best_eligible = (max(eligible_with_oracle_headroom,
                         key=lambda row: row["cleaned_image_mean_miou_empty0"])
                     if eligible_with_oracle_headroom else None)
    gates = {"one_confidence_rule_passes_oracle_fp_quality_cleaned_gain_and_beats_size_only":
             bool(eligible_with_oracle_headroom)}
    result = {
        "model": 370,
        "audit": "Model329 grouped train-OOF Eutectic component separability; diagnostic only",
        "metric": "image-wise four-class mIoU; empty union=0",
        "class_order": list(CLASS_ORDER),
        "oof_images": len(names),
        "specimen_groups": len(set(groups)),
        "input_scope": "By default, Model329 train-OOF probabilities and data/train/masks only; no valid/test paths are opened. --oof can override the probability archive, but targets still come only from data/train/masks.",
        "baseline_image_mean_miou_empty0": baseline,
        "candidate_rules": {"max_component_area_px": list(COMPONENT_SIZES),
                             "max_imagewise_margin_quantile": list(MARGIN_QUANTILES),
                             "remove_if": "predicted Eutectic component area <= A and component mean log(p_Eut)-max(log(p_other)) <= the imagewise q quantile among predicted Eutectic pixels",
                             "q_1_0_is_size_only_control": True,
                             "reassignment": "per-pixel argmax among the other three classes"},
        "component_margin_separability_by_size": component_margin_auc(
            component_auc_samples),
        "component_margin_auc_definition": "Within each component-size bin, compare low-margin scores (-component mean-margin imagewise quantile) for mostly-FP components (FP fraction >=0.80) against mostly-TP components (FP fraction <=0.20); intermediate components excluded; Mann-Whitney AUC, None if either group is empty.",
        "best_oracle_rule": best_oracle,
        "best_size_only_rule": max(
            size_only_by_area.values(),
            key=lambda row: row["cleaned_image_mean_miou_empty0"]),
        "best_in_sample_rule_passing_all_same_rule_gates": best_eligible,
        "gates": gates,
        "decision": "continue_to_train_only_stability_audit"
        if all(gates.values()) else "no_go_for_confidence_based_component_cleanup",
        "interpretation_caveat": "The 30 rules comprise 25 confidence-based rules (q<1) and 5 same-area size-only controls (q=1). Pixel purity/coverage/runner-up rates are pooled across pixels; cleaned and oracle mIoU are means over images. All rules are compared on the same grouped OOF images, so the in-sample cleaned delta is optimistic and not a CV estimate. A train-OOF source is not the exact Model360 route; any positive result needs a group-held-out stability audit before transfer.",
        "rules": records,
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "baseline_image_mean_miou_empty0", "best_oracle_rule",
        "best_in_sample_rule_passing_all_same_rule_gates", "gates", "decision")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
