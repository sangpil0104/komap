#!/usr/bin/env python3
"""Per-image/class audit for a single corrected restart checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask


def one_record(prediction: np.ndarray, target: np.ndarray) -> dict:
    values = []
    for class_id in range(len(CLASS_NAMES)):
        intersection = np.logical_and(prediction == class_id, target == class_id).sum()
        union = np.logical_or(prediction == class_id, target == class_id).sum()
        values.append(float(intersection / max(union, 1)))
    return {"image_mean_miou": float(np.mean(values)),
            "iou_internal": values,
            "iou_by_class": {name: value for name, value in zip(CLASS_NAMES, values)},
            "target_counts": [int(np.sum(target == class_id)) for class_id in range(len(CLASS_NAMES))],
            "prediction_counts": [int(np.sum(prediction == class_id)) for class_id in range(len(CLASS_NAMES))],
            "error_rate": float(np.mean(prediction != target)),
            "error_pixels": int(np.sum(prediction != target)),
            "pixels": int(target.size)}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    records = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.stride,
                                               device, use_amp, args.tile_batch_size, True)
        logits = np.log(np.maximum(probabilities, 1e-7))
        logits[3] += args.eutectic_bias
        prediction = logits.argmax(axis=0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        record = {"name": image_path.name, **one_record(prediction, target)}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result = {"checkpoint": args.checkpoint, "patch_size": args.patch_size,
              "stride": args.stride, "eutectic_bias": args.eutectic_bias,
              "image_mean_miou": float(np.mean([item["image_mean_miou"] for item in records])),
              "records": records}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image_mean_miou": result["image_mean_miou"],
                      "worst": sorted(records, key=lambda item: item["image_mean_miou"])[:5]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--eutectic-bias", type=float, default=-0.60)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
