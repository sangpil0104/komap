#!/usr/bin/env python3
"""Read-only audit of A71 under the fixed Model346 and Model349 routes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root))
    import model346_model336_into_341_route as model346  # noqa: PLC0415

    outputs = root / "outputs"
    model339 = json.loads((outputs / "model339_joint_gamma_route_search/results.json").read_text())
    model341 = json.loads((outputs / "model341_model329_joint_route_search/results.json").read_text())
    model346_result = json.loads((outputs / "model346_model336_into_341_route/results.json").read_text())
    model349_result = json.loads((outputs / "model349_model347_into_346_route/results.json").read_text())
    base_route = model339["best_image_empty_zero"]
    gamma329 = tuple(model341["best_image_empty_zero"]["gamma329"])
    gamma336 = tuple(model346_result["raw_best_gamma336"])
    gamma347 = tuple(model349_result["raw_best_gamma347"])
    expected_name = "A71_step2_sample1"
    source_paths = {
        "195": (outputs / "model195_native_resume/valid_probs.npz", False),
        "217": (outputs / "model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (outputs / "model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz", True),
        "336": (outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz", True),
        "347": (outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz", True),
    }
    archives = {}
    for source, (path, team_order) in source_paths.items():
        _, values = model346.model341.route.load_archive(path, team_order=team_order)
        archives[source] = values
        if expected_name not in values:
            raise ValueError(f"A71 missing from source {source}")

    names = sorted(archives["347"])
    if len(names) != 20 or names != model349_result["image_names"]:
        raise ValueError("Model349 image list differs from probability archives")
    index = names.index(expected_name)
    with (outputs / "model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    item = entries[index]
    mask_path = root / "data/valid/masks" / f"{expected_name}_mask.png"
    target = model346.decode_mask(Image.open(mask_path).convert("RGB"))
    if not np.array_equal(item["target"], target):
        raise ValueError("A71 route-cache target differs from official valid mask")
    sources = {source: values[expected_name] for source, values in archives.items()}
    sources["b4mean"] = 0.5 * (sources["322"] + sources["326"])
    row = {"index": index, "item": item,
           "base": model346.model341.route.undo_model181_primary_blend(item),
           "sources": {"base": None, "181": item["extra"].astype(np.float32), **sources}}
    guide = model346.model341.route.bilateral_weights(item["z"])

    def measure(which: str, gamma_347: tuple[float, ...]) -> dict:
        pred = model346._candidate_prediction(
            row, (0.0, 0.0, 0.0, 0.0), base_route,
            gamma329, guide, gamma336, gamma_347,
        )
        matrix = np.bincount(target.astype(np.int64).ravel() * 4 + pred.astype(np.int64).ravel(),
                             minlength=16).reshape(4, 4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
        ious = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
        return {"model": which, "image_miou_empty0": float(ious.mean()),
                "class_order": list(model346.CLASS_ORDER),
                "class_ious_empty0": ious.tolist(),
                "class_unions": union.tolist(),
                "primary_si_gt_pixels": int(np.count_nonzero(target == 1)),
                "primary_si_pred_pixels": int(np.count_nonzero(pred == 1))}

    print(json.dumps({"image_id": expected_name,
                      "model346": measure("346", (0.0, 0.0, 0.0, 0.0)),
                      "model349": measure("349", gamma347)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
