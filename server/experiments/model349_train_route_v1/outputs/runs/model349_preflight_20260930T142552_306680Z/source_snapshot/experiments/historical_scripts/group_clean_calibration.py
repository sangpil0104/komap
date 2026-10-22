#!/usr/bin/env python3
"""Calibrate class biases on validation images with no A-ID overlap with train."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import class_conditional, ensemble_probability, metrics, predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask, specimen_id


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    specs = [
        (args.checkpoint98, 512, 384, 4),
        (args.checkpoint99, 512, 384, 4),
        (args.checkpoint101, 640, 512, 2),
    ]
    models = [load_model(checkpoint, device) for checkpoint, _, _, _ in specs]
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    train_groups = {specimen_id(image_path) for image_path, _ in train_samples}
    cached = []
    for image_path, mask_path in _local_progress(valid_samples, 'group_clean_calibration'):
        probabilities = []
        for model, (_, patch_size, stride, batch_size) in zip(models, specs):
            probabilities.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                       batch_size, "d4", "gaussian"))
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append({"name": image_path.name, "group": specimen_id(image_path),
                       "probabilities": probabilities, "target": target})
        print(f"processed {image_path.name}", flush=True)
    clean = [item for item in cached if item["group"] not in train_groups]
    print(f"group_clean={len(clean)}/{len(cached)} images", flush=True)

    records = []
    for primary_bias in (-0.10, 0.0, 0.10):
        for eutectic_bias in (0.0, 0.10, 0.20):
            for al_bias in (-0.30, -0.20, -0.10):
                biases = np.asarray([0.0, primary_bias, al_bias, eutectic_bias], dtype=np.float32)[:, None, None]
                clean_predictions, clean_targets = [], []
                all_predictions, all_targets = [], []
                for item in cached:
                    combined = class_conditional(item["probabilities"])
                    prediction = (np.log(np.maximum(combined, 1e-7)) + biases).argmax(axis=0).astype(np.uint8)
                    all_predictions.append(prediction)
                    all_targets.append(item["target"])
                    if item["group"] not in train_groups:
                        clean_predictions.append(prediction)
                        clean_targets.append(item["target"])
                clean_metrics = metrics(clean_predictions, clean_targets)
                all_metrics = metrics(all_predictions, all_targets)
                records.append({"primary_bias": primary_bias, "eutectic_bias": eutectic_bias,
                                "al_bias": al_bias, "clean": clean_metrics, "all": all_metrics})
    best_clean = max(records, key=lambda x: x["clean"]["image_miou_compat"])
    best_all = max(records, key=lambda x: x["all"]["image_miou_compat"])
    result = {"train_groups": sorted(train_groups), "clean_images": [x["name"] for x in clean],
              "best_clean": best_clean, "best_all": best_all, "records": records}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_clean": best_clean, "best_all": best_all}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
