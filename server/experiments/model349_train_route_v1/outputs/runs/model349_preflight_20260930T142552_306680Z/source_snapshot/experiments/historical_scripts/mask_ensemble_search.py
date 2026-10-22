#!/usr/bin/env python3
"""Evaluate simple class-wise selection between two saved valid masks."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from PIL import Image

from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


def main(args):
    current_dir = Path(args.current_dir)
    other_dir = Path(args.other_dir)
    predictions, other_predictions, targets = [], [], []
    names = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        stem = image_path.name.replace("_image.png", "_mask.png")
        current = decode_mask(Image.open(current_dir / stem).convert("RGB"))
        other = decode_mask(Image.open(other_dir / stem).convert("RGB"))
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(current)
        other_predictions.append(other)
        targets.append(target)
        names.append(stem)
    records = []
    for source_bits in product((0, 1), repeat=4):
        chosen = []
        for current, other in zip(predictions, other_predictions):
            prediction = current.copy()
            for class_id, use_other in enumerate(source_bits):
                if use_other:
                    prediction[other == class_id] = class_id
            chosen.append(prediction)
        result = metrics(chosen, targets)
        records.append({"other_for_internal_classes": source_bits, **result})
    records.extend([
        {"source": "current", **metrics(predictions, targets)},
        {"source": "other", **metrics(other_predictions, targets)},
    ])
    best = max(records, key=lambda item: item["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--current-dir", required=True)
    parser.add_argument("--other-dir", required=True)
    parser.add_argument("--output-json", required=True)
    main(parser.parse_args())
