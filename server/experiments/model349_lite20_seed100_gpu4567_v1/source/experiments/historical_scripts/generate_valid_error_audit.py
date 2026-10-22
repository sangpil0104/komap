#!/usr/bin/env python3
"""Generate model57 validation predictions, ground truth and binary error maps."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from ensemble_refinements import combine_with_new, load_aspp, load_refined
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask, encode_mask


def per_image_metrics(prediction, target):
    values = []
    for class_id in range(4):
        intersection = np.logical_and(prediction == class_id, target == class_id).sum()
        union = np.logical_or(prediction == class_id, target == class_id).sum()
        values.append(float(intersection / max(union, 1)))
    return {"miou": float(np.mean(values)), "iou": values,
            "iou_by_class": {name: float(value) for name, value in zip(CLASS_NAMES, values)},
            "error_rate": float(np.mean(prediction != target)),
            "error_pixels": int(np.sum(prediction != target)),
            "pixels": int(target.size)}


def make_error_map(prediction, target):
    """Green = correct, red = misclassified; no class-name assumptions."""
    output = np.zeros((*target.shape, 3), dtype=np.uint8)
    output[:] = (25, 29, 36)
    output[prediction == target] = (38, 105, 76)
    output[prediction != target] = (232, 67, 67)
    return Image.fromarray(output, mode="RGB")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")

    # Keep only one network on the GPU at a time.  The five-network ensemble
    # otherwise exceeds a 16 GB P100 before the first validation image.
    model_specs = [
        ("model98", lambda: load_model(args.checkpoint98, device), 512, 384, 4),
        ("model99", lambda: load_model(args.checkpoint99, device), 512, 384, 4),
        ("model101", lambda: load_model(args.checkpoint101, device), 640, 512, 2),
        ("model109", lambda: load_aspp(args.checkpoint109, args.checkpoint98, device), 512, 384, 4),
        ("model123_ema", lambda: load_refined("ema", args.checkpoint123, args.checkpoint109, device), 512, 384, 4),
    ]
    cached_values = []
    for name, loader, patch_size, stride, batch_size in model_specs:
        print(f"loading {name}", flush=True)
        model = loader()
        values = []
        for image_path, _ in _local_progress(samples, 'generate_valid_error_audit'):
            values.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                batch_size, "d4", "gaussian"))
            print(f"  predicted {image_path.name}", flush=True)
        cached_values.append(values)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output_dir)
    for name in ("prediction", "ground_truth", "error_map"):
        (output / name).mkdir(parents=True, exist_ok=True)
    records = []
    all_predictions, all_targets = [], []
    bias = np.asarray([args.al3ni_bias, args.primary_bias, args.al_bias, args.eutectic_bias],
                      dtype=np.float32)[:, None, None]
    for sample_index, (image_path, mask_path) in enumerate(_local_progress(samples, 'generate_valid_error_audit')):
        base_values = [values[sample_index] for values in cached_values[:4]]
        ema_value = cached_values[4][sample_index]
        combined = combine_with_new(base_values, [ema_value], 0.25)
        prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        stem = image_path.name.replace("_image.png", "")
        encode_mask(prediction).save(output / "prediction" / f"{stem}_mask.png")
        encode_mask(target).save(output / "ground_truth" / f"{stem}_mask.png")
        make_error_map(prediction, target).save(output / "error_map" / f"{stem}_error.png")
        item = {"name": stem, "image": image_path.name, **per_image_metrics(prediction, target)}
        records.append(item)
        all_predictions.append(prediction)
        all_targets.append(target)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    summary = metrics(all_predictions, all_targets)
    manifest = {"model": "mapping-corrected model57 / model124 refined ensemble", "bias_official": {"Primary Si": args.primary_bias,
                "Al3Ni": args.al3ni_bias, "Eutectic Si": args.eutectic_bias, "Al": args.al_bias},
                "records": records, "global": summary}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(records), "global": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--checkpoint123", required=True)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--al3ni-bias", type=float, default=0.0)
    parser.add_argument("--eutectic-bias", type=float, default=0.10)
    parser.add_argument("--al-bias", type=float, default=-0.30)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
