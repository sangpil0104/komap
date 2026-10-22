#!/usr/bin/env python3
"""Export exact Model326+Model270 validation predictions for the visual audit.

Uses saved validation probabilities and the already-selected route. It performs
no model inference and never reads test data.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict
from model323_model270_branch_route_probe import (
    canonical_image_key,
    load_archive,
)


PALETTE = (
    (25, 76, 153),   # Al3Ni
    (76, 178, 76),   # Primary Si
    (153, 127, 76),  # Al
    (204, 204, 204), # Eutectic Si
)
CLASS_NAMES = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")
ERROR_CORRECT = (38, 105, 76)
ERROR_WRONG = (232, 67, 67)


def save_mask(values: np.ndarray, path: Path) -> None:
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        rgb[values == class_id] = color
    Image.fromarray(rgb, mode="RGB").save(path)


def score_image(prediction: np.ndarray, target: np.ndarray) -> dict:
    ious = []
    for class_id in range(4):
        gt = target == class_id
        pred = prediction == class_id
        union = int(np.count_nonzero(gt | pred))
        intersection = int(np.count_nonzero(gt & pred))
        ious.append(intersection / union if union else 0.0)
    wrong = prediction != target
    return {
        "miou": float(np.mean(ious)),
        "iou": ious,
        "iou_by_class": dict(zip(CLASS_NAMES, map(float, ious))),
        "error_rate": float(wrong.mean()),
        "error_pixels": int(wrong.sum()),
        "pixels": int(target.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    result_path = root / "outputs/model326_model270_fusion/route_probe.json"
    with result_path.open(encoding="utf-8") as handle:
        route_result = json.load(handle)
    setting = route_result["best_image_empty_zero"]
    candidate_id = route_result["candidate_id"]
    paths = {
        "195": root / "outputs/model195_native_resume/valid_probs.npz",
        "217": root / "outputs/model220_native_tent_extended/valid_probs_06.npz",
        "269": root / "outputs/model269_fullstats_1x_extended/valid_probs.npz",
        candidate_id: root / "outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz",
    }
    with (root / "outputs/model183_metric_calibration/entries.pkl").open("rb") as handle:
        entries = pickle.load(handle)

    archives: dict[str, dict[str, np.ndarray]] = {}
    image_names: list[str] | None = None
    for source, path in paths.items():
        team_order = (source == candidate_id
                      and route_result["candidate_order_input"] == "team")
        names, values = load_archive(path, team_order=team_order)
        normalized = {canonical_image_key(name): values[name] for name in names}
        if len(normalized) != len(names):
            raise ValueError(f"duplicate normalized image IDs in {path}")
        if image_names is None:
            image_names = [canonical_image_key(name) for name in names]
        elif set(normalized) != set(image_names):
            raise ValueError(f"validation image IDs differ in {path}")
        archives[source] = normalized
    if image_names is None or len(entries) != len(image_names):
        raise ValueError(f"entry/archive count mismatch: {len(entries)} vs {len(image_names or [])}")
    if len(image_names) != 20:
        raise ValueError(f"expected validation 20, got {len(image_names)}")

    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "index": index,
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: values[image_names[index]]
                   for source, values in archives.items()},
            },
        })

    for subdir in ("original", "ground_truth", "prediction", "error_map"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    records = []
    confusion = np.zeros((4, 4), dtype=np.int64)
    expected_scores = np.asarray(setting["image_scores"], dtype=np.float64)
    for row in rows:
        name = image_names[row["index"]]
        guide_weights = bilateral_weights(row["item"]["z"])
        mixed = make_mixed(row, setting)
        al_gamma = float(setting.get("al_gamma", 0.0))
        if al_gamma:
            al = row["sources"][candidate_id][2]
            mixed[2] = (1.0 - al_gamma) * mixed[2] + al_gamma * al
            mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        prediction = predict(row["item"], mixed, setting, guide_weights, exact=True)
        target = np.asarray(row["item"]["target"], dtype=np.uint8)
        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch for {name}: {prediction.shape} vs {target.shape}")

        values = score_image(prediction, target)
        if abs(values["miou"] - expected_scores[row["index"]]) > 1e-7:
            raise AssertionError(
                f"route score mismatch for {name}: {values['miou']} vs "
                f"{expected_scores[row['index']]}"
            )
        confusion += np.bincount(
            target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16
        ).reshape(4, 4)

        stem = name.removesuffix("_image")
        source_image = root / "data/valid/images" / f"{stem}_image.png"
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        shutil.copyfile(source_image, output / "original" / f"{stem}_image.png")
        save_mask(target, output / "ground_truth" / f"{stem}_mask.png")
        save_mask(prediction, output / "prediction" / f"{stem}_mask.png")
        error = np.empty((*target.shape, 3), dtype=np.uint8)
        error[:] = ERROR_WRONG
        error[prediction == target] = ERROR_CORRECT
        Image.fromarray(error, mode="RGB").save(output / "error_map" / f"{stem}_error.png")
        records.append({"name": stem, "image": f"{stem}_image.png", **values})

    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    global_iou = diagonal / np.maximum(union, 1.0)
    manifest = {
        "model": "Model326 EfficientNet-B4 + Model270 class-wise route",
        "metric": "per-image mean of 4 class IoUs; empty union=0",
        "route": {key: setting[key] for key in (
            "primary_source", "primary_gamma", "al3ni_source", "al3ni_gamma",
            "al_source", "al_gamma", "eutectic_source", "eutectic_gamma",
            "prior_scale", "beta_scale", "al_delta", "eut_delta",
            "gaussian_blend", "eut_edge_blend", "cleanup_threshold")},
        "records": records,
        "global": {
            "global_miou": float(global_iou.mean()),
            "global_iou": dict(zip(CLASS_NAMES, map(float, global_iou))),
            "image_mean_miou": float(np.mean([row["miou"] for row in records])),
            "confusion": confusion.tolist(),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "images": len(records),
        "image_mean_miou": manifest["global"]["image_mean_miou"],
        "global_miou": manifest["global"]["global_miou"],
        "route_expected_image_mean_miou": setting["image_mean_miou"],
        "route_exact_score_match": True,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
