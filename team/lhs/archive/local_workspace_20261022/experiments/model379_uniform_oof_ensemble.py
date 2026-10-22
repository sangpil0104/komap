#!/usr/bin/env python3
"""Evaluate the pre-registered fixed 1/3 Model374/376/377 OOF ensemble."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


STEP_RE = re.compile(r"_step(\d+)_")


def load_rows(cv_root: Path, model_roots: dict[str, Path]) -> dict[str, dict]:
    evaluator_root = Path("/content/komap_model1/model377_directional/experiments")
    sys.path.insert(0, str(evaluator_root))
    from model377_oof_evaluation import (  # type: ignore[import-not-found]
        GROUP_RE,
        class_ious,
        mask_to_ids,
    )

    rows = {"Model374": {}, "Model379": {}}
    seen: set[str] = set()
    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}/masks"
        image_dir = cv_root / f"oof_fold{fold}/images"
        masks = {path.name.removesuffix("_mask.png"): path
                 for path in mask_dir.glob("*_mask.png")}
        if not masks or not image_dir.is_dir():
            raise FileNotFoundError(f"missing fixed OOF data for fold {fold}")

        archives = {
            model: np.load(root / f"fold{fold}_oof/valid_probabilities.npz", allow_pickle=False)
            for model, root in model_roots.items()
        }
        try:
            expected = set(masks)
            for model, archive in archives.items():
                if set(archive.files) != expected:
                    raise ValueError(f"fold {fold}: {model} image IDs do not match OOF masks")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image: {image_id}")
                seen.add(image_id)
                group_match = GROUP_RE.match(image_id)
                step_match = STEP_RE.search(image_id)
                if not group_match or not step_match:
                    raise ValueError(f"cannot parse specimen/step from {image_id}")

                truth = mask_to_ids(masks[image_id])
                probabilities = {
                    model: np.asarray(archive[image_id], dtype=np.float32)
                    for model, archive in archives.items()
                }
                reference = probabilities["Model374"]
                if reference.shape != (4, *truth.shape):
                    raise ValueError(f"fold {fold}: invalid probability shape for {image_id}")
                for model, value in probabilities.items():
                    if value.shape != reference.shape or not np.isfinite(value).all():
                        raise ValueError(f"fold {fold}: invalid {model} probability map for {image_id}")
                    if np.max(np.abs(value.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"fold {fold}: {model} probabilities are not normalized")

                ensemble = (probabilities["Model374"] + probabilities["Model376"]
                            + probabilities["Model377"]) / 3.0
                with Image.open(image_dir / f"{image_id}_image.png") as image:
                    resolution = f"{image.width}x{image.height}"
                for model, value in (("Model374", reference), ("Model379", ensemble)):
                    prediction = value.argmax(axis=0)
                    rows[model][image_id] = {
                        "fold": fold,
                        "group": group_match.group(1),
                        "step": int(step_match.group(1)),
                        "resolution": resolution,
                        "class_iou": class_ious(value, truth),
                        "true_class_fraction": np.asarray(
                            [(truth == class_id).mean() for class_id in range(4)], dtype=np.float64
                        ),
                        "predicted_class_fraction": np.asarray(
                            [(prediction == class_id).mean() for class_id in range(4)], dtype=np.float64
                        ),
                    }
        finally:
            for archive in archives.values():
                archive.close()

    if len(seen) != 70 or len(set(row["group"] for row in rows["Model374"].values())) != 59:
        raise ValueError(f"expected 70 images/59 groups exactly once; got {len(seen)}")
    return rows


def run(cv_root: Path, roots: dict[str, Path], draws: int) -> dict:
    evaluator_root = Path("/content/komap_model1/model377_directional/experiments")
    sys.path.insert(0, str(evaluator_root))
    from model377_oof_evaluation import compare_miou, summarize_miou  # type: ignore[import-not-found]

    rows = load_rows(cv_root, roots)
    scores = {model: summarize_miou(rows, model, draws) for model in rows}
    comparison = compare_miou(rows["Model374"], rows["Model379"], draws, 379)
    step4_names = [name for name, row in rows["Model374"].items() if row["step"] == 4]
    if not step4_names:
        raise ValueError("no Step4 OOF images found")
    step4_reference = float(np.mean([
        np.mean(rows["Model374"][name]["class_iou"]) for name in step4_names
    ]) * 100)
    step4_ensemble = float(np.mean([
        np.mean(rows["Model379"][name]["class_iou"]) for name in step4_names
    ]) * 100)
    step4_delta = step4_ensemble - step4_reference

    gate = {
        "gain_at_least_0_5pp": comparison["delta_pp"] >= 0.5,
        "group_bootstrap_ci_lower_above_zero":
            comparison["paired_group_bootstrap_95ci_delta_pp"][0] > 0,
        "at_least_3_of_4_folds_positive":
            sum(delta > 0 for delta in comparison["fold_delta_pp"].values()) >= 3,
        "worst_fold_at_least_minus_0_3pp": min(comparison["fold_delta_pp"].values()) >= -0.3,
        "no_class_mean_iou_drop_over_0_5pp":
            all(delta >= -0.5 for delta in comparison["class_mean_iou_delta_pp"].values()),
        "step4_delta_at_least_minus_0_5pp": step4_delta >= -0.5,
    }
    return {
        "model": 379,
        "scope": "70 training images, exactly-once specimen-grouped OOF; valid/test excluded",
        "recipe": "fixed arithmetic mean of same-fold Model374/376/377 probabilities, 1/3 each; argmax",
        "metric": "per-image arithmetic mean of four class IoUs; empty union=0",
        "images": 70,
        "specimen_groups": 59,
        "model_scores": scores,
        "comparison_vs_model374": comparison,
        "step4_subset": {
            "images": len(step4_names),
            "specimen_groups": len({rows["Model374"][name]["group"] for name in step4_names}),
            "model374_image_mean_miou_pct": step4_reference,
            "model379_image_mean_miou_pct": step4_ensemble,
            "delta_pp": step4_delta,
        },
        "adoption_gate": gate,
        "adoption_gate_passed": all(gate.values()),
        "interpretation_guard": (
            "This fixed ensemble is conditional on prior recipe/model selection; OOF is not an independent test. "
            "No ensemble weights, class calibration, TTA, routing, morphology, valid, or test labels were tuned."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--model376-root", type=Path, required=True)
    parser.add_argument("--model377-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cv_root, {
        "Model374": args.model374_root,
        "Model376": args.model376_root,
        "Model377": args.model377_root,
    }, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": result["model"],
        "model_scores": result["model_scores"],
        "delta_vs_374_pp": result["comparison_vs_model374"]["delta_pp"],
        "delta_ci95_pp": result["comparison_vs_model374"]["paired_group_bootstrap_95ci_delta_pp"],
        "fold_deltas_pp": result["comparison_vs_model374"]["fold_delta_pp"],
        "class_deltas_pp": result["comparison_vs_model374"]["class_mean_iou_delta_pp"],
        "step4_subset": result["step4_subset"],
        "adoption_gate": result["adoption_gate"],
        "adoption_gate_passed": result["adoption_gate_passed"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
