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

from komap_seg.checkpoint import load_model
from komap_seg.dataset import KomapFullDataset
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
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=170)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root) / args.split
    masks = root / "masks" if args.split == "valid" else None
    ds = KomapFullDataset(root / "images", masks)
    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), device)

    probabilities: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for sample in ds:
        probs = predict_probs_tta(
            model,
            sample["image"],
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            tta=not args.no_tta,
            input_scale=args.input_scale,
            device=device,
        ).numpy().astype(np.float32)
        probabilities[sample["stem"]] = probs.astype(np.float16)
        pred = probs.argmax(axis=0)
        row: dict[str, object] = {"image_id": sample["stem"]}
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
            bins = target.reshape(-1) * NUM_CLASSES + pred.reshape(-1)
            pooled += np.bincount(bins, minlength=NUM_CLASSES**2).reshape(
                NUM_CLASSES, NUM_CLASSES
            )
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
            "patch_size": args.patch_size,
            "input_scale": args.input_scale,
            "overlap": args.overlap,
            "tta": "4-way horizontal/vertical flips" if not args.no_tta else "none",
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
        intersection = np.diag(pooled).astype(np.float64)
        union = pooled.sum(0) + pooled.sum(1) - intersection
        global_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        summary["pooled_global_miou_diagnostic_only"] = float(global_iou.mean())
        summary["pooled_global_iou_by_class"] = {
            name: float(value) for name, value in zip(CLASS_NAMES, global_iou)
        }
    (output / f"{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
