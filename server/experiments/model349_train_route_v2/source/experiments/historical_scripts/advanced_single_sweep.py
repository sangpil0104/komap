#!/usr/bin/env python3
"""Sweep inference-time TTA/tile blending for one restart checkpoint."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    samples = collect_split(args.data_dir, "valid")
    all_results = []
    for tta_mode in ("legacy", "d4"):
        for blend_mode in ("uniform", "gaussian"):
            cached = []
            for image_path, mask_path in _local_progress(samples, 'advanced_single_sweep'):
                probability = predict_probabilities(
                    model, image_path, args.patch_size, args.stride, device, use_amp,
                    args.tile_batch_size, tta_mode, blend_mode)
                cached.append((probability, decode_mask(Image.open(mask_path).convert("RGB"))))
                print(f"{tta_mode}/{blend_mode} cached {image_path.name}", flush=True)
            for bias in np.arange(-0.80, 0.1001, 0.05):
                predictions, targets = [], []
                for probability, target in cached:
                    logits = np.log(np.maximum(probability, 1e-7))
                    logits[3] += float(bias)
                    predictions.append(logits.argmax(axis=0).astype(np.uint8))
                    targets.append(target)
                result = metrics(predictions, targets)
                all_results.append({"tta": tta_mode, "blend": blend_mode,
                                    "eutectic_bias": float(bias), **result})
            best_mode = max((x for x in all_results if x["tta"] == tta_mode and x["blend"] == blend_mode),
                            key=lambda x: x["image_mean_miou"])
            print(json.dumps({"best_mode": best_mode}, ensure_ascii=False), flush=True)
    best = max(all_results, key=lambda x: x["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
