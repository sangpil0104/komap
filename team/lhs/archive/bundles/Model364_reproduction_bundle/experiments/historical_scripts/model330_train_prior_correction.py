#!/usr/bin/env python3
"""Apply the teammate's train-pixel class-weight correction to Model327's best route."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from model183_metric_calibration import prior_terms
from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import (
    apply_bilateral,
    bilateral_weights,
    make_mixed,
)
from model323_model270_branch_route_probe import (
    TEAM_TO_MAIN,
    canonical_image_key,
    load_archive,
)


PALETTE_TEAM = np.array([
    [153, 127, 76],   # Al
    [25, 76, 153],    # Al3Ni
    [76, 178, 76],    # Primary Si
    [204, 204, 204],  # Eutectic Si
], dtype=np.uint8)
SOURCES = {
    "195": "outputs/model195_native_resume/valid_probs.npz",
    "217": "outputs/model220_native_tent_extended/valid_probs_06.npz",
    "269": "outputs/model269_fullstats_1x_extended/valid_probs.npz",
    "team_b4_model322": "outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz",
}


def train_class_weights(mask_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(PALETTE_TEAM), dtype=np.int64)
    files = sorted(mask_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no training masks in {mask_dir}")
    for path in files:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        counts += np.array([np.count_nonzero(np.all(rgb == color, axis=2))
                            for color in PALETTE_TEAM], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError(f"missing class pixels in training set: {counts.tolist()}")
    # Reciprocal pixel prevalence, exactly as described by the teammate.
    prevalence = counts / counts.sum()
    return counts, 1.0 / prevalence


def refined_logits(row: dict, setting: dict, guide_weights: list[np.ndarray]) -> np.ndarray:
    """Reproduce Model198's exact route through spatial refinement, before argmax."""
    item = row["item"]
    mixed = make_mixed(row, setting)
    al_gamma = float(setting.get("al_gamma", 0.0))
    if al_gamma:
        al = row["sources"]["team_b4_model322"][2]
        mixed[2] = (1.0 - al_gamma) * mixed[2] + al_gamma * al
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)

    base_prior, delta = prior_terms(item, mixed)
    prior = setting["prior_scale"] * base_prior
    prior[0] += (setting["prior_scale"] * setting["beta_scale"]
                  * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + setting["al_delta"]
    logits[3] += item["params"]["eut_bias"] + setting["eut_delta"]

    refined = logits.copy()
    refined[1] = (setting["gaussian_blend"] * gaussian_filter(
        logits[1], sigma=0.50, mode="reflect")
        + (1.0 - setting["gaussian_blend"])
        * apply_bilateral(logits[1], guide_weights))
    refined[3] = ((1.0 - setting["eut_edge_blend"]) * logits[3]
                  + setting["eut_edge_blend"]
                  * apply_bilateral(logits[3], guide_weights))
    return refined


def postprocess(refined: np.ndarray, offsets: np.ndarray,
                cleanup_threshold: int) -> np.ndarray:
    # Dividing softmax probabilities by class weights is equivalent to adding
    # -log(weight) to the corresponding logits immediately before argmax.
    adjusted = refined + offsets[:, None, None]
    prediction = cleanup_primary(adjusted.argmax(axis=0).astype(np.uint8),
                                 adjusted, cleanup_threshold)
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = adjusted.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
    return prediction


def summarize(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    confusion = np.zeros((4, 4), dtype=np.int64)
    per_image = []
    per_image_exclude = []
    class_image_iou = []
    for pred, target in zip(predictions, targets):
        cm = np.bincount(target.reshape(-1).astype(np.int64) * 4 + pred.reshape(-1),
                         minlength=16).reshape(4, 4)
        confusion += cm
        inter = np.diag(cm).astype(np.float64)
        union = cm.sum(0) + cm.sum(1) - inter
        iou = inter / np.maximum(union, 1.0)  # empty union scores zero
        per_image.append(float(iou.mean()))
        per_image_exclude.append(float(iou[union > 0].mean()))
        class_image_iou.append(iou)
    inter = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - inter
    class_names = ["Al3Ni", "Primary_Si", "Al", "Eutectic_Si"]
    return {
        "image_mean_miou_empty_zero": float(np.mean(per_image)),
        "image_mean_miou_exclude_empty": float(np.mean(per_image_exclude)),
        "image_mean_iou_per_class_empty_zero": {
            name: float(x) for name, x in zip(class_names,
                                              np.mean(class_image_iou, axis=0))},
        "global_iou_per_class": {
            name: float(x) for name, x in zip(class_names,
                                              inter / np.maximum(union, 1))},
        "images": len(predictions),
        "confusion": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/model330_train_prior_correction/result.json"))
    args = parser.parse_args()
    root = args.root

    counts, class_weights_team = train_class_weights(root / "data/train/masks")
    weights_main = class_weights_team[list(TEAM_TO_MAIN)]
    offsets = -np.log(weights_main)

    with (root / "outputs/model183_metric_calibration/entries.pkl").open("rb") as f:
        entries = pickle.load(f)
    archives = {}
    image_names = None
    for source, relative in SOURCES.items():
        path = root / relative
        team_order = source == "team_b4_model322"
        names, values = load_archive(path, team_order=team_order)
        normalized = {canonical_image_key(k): values[k] for k in names}
        if image_names is None:
            image_names = [canonical_image_key(k) for k in names]
        elif set(normalized) != set(image_names):
            raise ValueError(f"validation image mismatch in {path}")
        archives[source] = normalized
    if len(entries) != len(image_names or []):
        raise ValueError(f"entries={len(entries)}, probability images={len(image_names or [])}")

    result327 = json.loads((root / "outputs/model327_b4_classwise_route/route_probe.json")
                           .read_text(encoding="utf-8"))
    setting = {k: v for k, v in result327["best_image_empty_zero"].items()
               if k not in {"candidate", "global_miou", "global_iou", "image_mean_miou",
                            "image_mean_miou_exclude", "image_scores", "image_scores_exclude",
                            "confusion"}}
    predictions_raw, predictions_corrected, targets = [], [], []
    for index, item in enumerate(entries):
        name = image_names[index]
        row = {
            "index": index,
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{source: archives[source][name] for source in archives}},
        }
        refined = refined_logits(row, setting, bilateral_weights(item["z"]))
        predictions_raw.append(postprocess(refined, np.zeros(4),
                                           int(setting["cleanup_threshold"])))
        predictions_corrected.append(postprocess(refined, offsets,
                                                 int(setting["cleanup_threshold"])))
        targets.append(item["target"])
        print(f"processed {index + 1}/{len(entries)}: {name}", flush=True)

    report = {
        "method": "divide final route softmax probabilities by reciprocal train-mask pixel prevalence before argmax",
        "class_order_team": ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"],
        "class_order_route": ["Al3Ni", "Primary_Si", "Al", "Eutectic_Si"],
        "train_mask_count": len(list((root / "data/train/masks").glob("*.png"))),
        "train_pixel_counts_team_order": counts.tolist(),
        "train_pixel_prevalence_team_order": (counts / counts.sum()).tolist(),
        "class_weights_inverse_prevalence_team_order": class_weights_team.tolist(),
        "class_weights_inverse_prevalence_route_order": weights_main.tolist(),
        "logit_offsets_route_order": offsets.tolist(),
        "route_candidate": result327["best_image_empty_zero"]["candidate"],
        "recorded_model327_miou": result327["best_image_empty_zero"]["image_mean_miou"],
        "uncorrected_reproduction": summarize(predictions_raw, targets),
        "corrected": summarize(predictions_corrected, targets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
