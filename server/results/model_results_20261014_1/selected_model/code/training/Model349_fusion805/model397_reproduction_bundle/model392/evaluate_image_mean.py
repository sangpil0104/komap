"""Official per-image four-class mIoU evaluation for Model336.

The upstream HRNet_best evaluate.py reports a single pooled confusion-matrix
score. This evaluator preserves its inference path but additionally computes
the competition's mean of per-image four-class IoUs, including explicit
empty-union policies, and saves per-image probability maps for later audits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
from komap_seg.inference import predict_probs_tta
from komap_seg.palette import CLASS_NAMES, NUM_CLASSES


def class_ious(pred: np.ndarray, target: np.ndarray) -> tuple[list[float | None], list[int]]:
    values: list[float | None] = []
    unions: list[int] = []
    for class_id in range(NUM_CLASSES):
        truth = target == class_id
        guess = pred == class_id
        union = int(np.logical_or(truth, guess).sum())
        intersection = int(np.logical_and(truth, guess).sum())
        unions.append(union)
        values.append(intersection / union if union else None)
    return values, unions


def means(values: list[float | None]) -> dict[str, float]:
    present = [value for value in values if value is not None]
    return {
        "empty_zero": float(sum(present) / NUM_CLASSES),
        "empty_skip": float(sum(present) / len(present)) if present else float("nan"),
        "empty_one": float((sum(present) + NUM_CLASSES - len(present)) / NUM_CLASSES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=170)
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument(
        "--target-width", type=int, default=None,
        help="Optional assertion against checkpoint metadata; by default use the saved width.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta-mode", choices=("flip4", "d4"), default="flip4",
                        help="flip4 preserves Model336; d4 adds 90-degree rotations")
    parser.add_argument("--downsample-antialias", action="store_true",
                        help="low-pass filter probability maps when reducing scaled tiles")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root) / args.split
    masks = root / "masks" if args.split == "valid" else None
    ds = KomapFullDataset(root / "images", masks)
    device = torch.device(args.device)
    model, saved_preprocessing = load_model_and_preprocessing(Path(args.checkpoint), device)
    preprocessing = resolve_preprocessing(
        saved_preprocessing, args.patch_size, args.input_scale, args.target_width
    )

    probabilities: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for sample in ds:
        sample_scale = effective_input_scale(
            preprocessing["input_scale"], preprocessing["target_width"], sample["width"]
        )
        sample_overlap = scaled_overlap(
            preprocessing["patch_size"], preprocessing["input_scale"], args.overlap, sample_scale
        )
        probs = predict_probs_tta(
            model,
            sample["image"],
            patch_size=preprocessing["patch_size"],
            overlap=sample_overlap,
            batch_size=args.batch_size,
            tta=not args.no_tta,
            input_scale=sample_scale,
            device=device,
            tta_mode=args.tta_mode,
            downsample_antialias=args.downsample_antialias,
        ).numpy().astype(np.float32)
        probabilities[sample["stem"]] = probs.astype(np.float16)
        pred = probs.argmax(axis=0)
        row: dict[str, object] = {
            "image_id": sample["stem"],
            "native_width": sample["width"],
            "effective_input_scale": sample_scale,
            "effective_overlap": sample_overlap,
        }
        if args.split == "valid":
            target = sample["mask"].numpy()
            ious, unions = class_ious(pred, target)
            image_means = means(ious)
            row.update({
                "image_miou_empty_zero": image_means["empty_zero"],
                "image_miou_empty_skip": image_means["empty_skip"],
                "image_miou_empty_one": image_means["empty_one"],
                "empty_class_count": sum(union == 0 for union in unions),
            })
            for class_id, name in enumerate(CLASS_NAMES):
                row[f"iou_{name}"] = ious[class_id]
        rows.append(row)
        if args.split == "valid":
            print(
                f"{sample['stem']}: image mIoU zero="
                f"{image_means['empty_zero']:.6f}, skip="
                f"{image_means['empty_skip']:.6f}",
                flush=True,
            )
        else:
            print(f"{sample['stem']}: predicted", flush=True)

    np.savez_compressed(output / f"{args.split}_probabilities.npz", **probabilities)
    with (output / f"{args.split}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "model": "Model336 HRNet-W48 + SMP U-Net++",
        "split": args.split,
        "images": len(ds),
        "class_order": CLASS_NAMES,
        "inference": {
            "patch_size": preprocessing["patch_size"],
            "base_input_scale": preprocessing["input_scale"],
            "target_width": preprocessing["target_width"],
            "scale_rule": (
                "base_input_scale * target_width / native image width"
                if preprocessing["target_width"] else "fixed base_input_scale"
            ),
            "overlap": args.overlap,
            "overlap_rule": (
                "preserve overlap/native-tile ratio"
                if preprocessing["target_width"] else "fixed"
            ),
            "tta": ("none" if args.no_tta else
                    "4-way horizontal/vertical flips" if args.tta_mode == "flip4" else
                    "8-way dihedral rotations and flips"),
            "tta_mode": args.tta_mode,
            "tta_enabled": not args.no_tta,
            "downsample_antialias": args.downsample_antialias,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "per_image_metrics": str(output / f"{args.split}_per_image.csv"),
        "probabilities": str(output / f"{args.split}_probabilities.npz"),
    }
    if args.split == "valid":
        for policy, field in (("empty_zero", "image_miou_empty_zero"),
                              ("empty_skip", "image_miou_empty_skip"),
                              ("empty_one", "image_miou_empty_one")):
            vals = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
            summary[field] = float(np.mean(vals))
    (output / f"{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
