#!/usr/bin/env python3
"""Measure practical and oracle image-wise routing ceilings.

The router uses only prediction probabilities and image-level statistics at
inference time.  The oracle sections are diagnostic only: they use validation
labels to show how much headroom remains if a perfect per-image selector were
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)
BIASES = np.asarray([0.0, 0.0, 0.20, -0.55], dtype=np.float32)


def primary_cleanup(prediction: np.ndarray, logits: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Remove tiny Primary-Si components and assign their best other logit."""
    from scipy import ndimage

    mask = prediction == 1
    labels, count = ndimage.label(mask)
    if count == 0:
        return prediction
    sizes = np.bincount(labels.reshape(-1))
    small = mask & (sizes[labels] <= threshold)
    if not small.any():
        return prediction
    output = prediction.copy()
    other = logits.copy()
    other[1] = -np.inf
    output[small] = other[:, small].argmax(axis=0).astype(np.uint8)
    return output


def score_one(prediction: np.ndarray, target: np.ndarray) -> float:
    current = np.bincount(
        target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16
    ).reshape(4, 4)
    inter = np.diag(current).astype(np.float64)
    union = current.sum(0) + current.sum(1) - inter
    return float(np.mean(inter / np.maximum(union, 1.0)))


def make_base(values: list[np.ndarray], gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mixed = np.asarray([
        sum(MATRIX[class_id, model_id] * values[model_id][class_id]
            for model_id in range(3))
        for class_id in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
    logits = np.log(np.maximum(mixed, 1e-7))
    logits += BIASES[:, None, None]
    prediction = logits.argmax(axis=0).astype(np.uint8)
    return logits, prediction


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = build_model("aspp", pretrained=False, checkpoint=args.aspp_base_checkpoint).to(device)
    aspp.load_state_dict(torch.load(args.aspp_checkpoint, map_location="cpu")["model"])
    aspp.eval()
    models = (old, new, aspp)
    samples = collect_split(args.data_dir, "valid")
    records = []
    cache_dir = Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in samples:
        cache_path = cache_dir / f"{image_path.stem}.npz"
        if cache_path.exists():
            loaded = np.load(cache_path)
            values = [loaded[f"p{i}"] for i in range(3)]
        else:
            values = [
                predict_probabilities(model, image_path, patch_size, stride, device, use_amp, 2, True)
                for model, patch_size, stride in zip(models, (640, 512, 512), (512, 384, 384))
            ]
            np.savez_compressed(cache_path, **{f"p{i}": value.astype(np.float16)
                                               for i, value in enumerate(values)})
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        base_logits, base_prediction = make_base(values, gray)
        base_prediction = primary_cleanup(base_prediction, base_logits, 128)
        candidates = [
            np.log(np.maximum(value, 1e-7)).argmax(axis=0).astype(np.uint8)
            for value in values
        ]
        candidates.append(base_prediction)
        scores = [score_one(candidate, target) for candidate in candidates]

        # A practical, label-free router candidate: select the source with the
        # lowest mean pixel entropy.  It is deliberately simple and cannot use
        # validation labels at inference time.
        entropies = [float((-value * np.log(np.maximum(value, 1e-7))).sum(axis=0).mean())
                     for value in values]
        route_index = int(np.argmin(entropies))
        route_prediction = candidates[route_index]
        records.append({
            "name": image_path.name,
            "scores": scores,
            "entropy": entropies,
            "predicted_fraction": [np.bincount(candidate.reshape(-1), minlength=4).tolist()
                                    for candidate in candidates],
            "route_index": route_index,
            "route_score": score_one(route_prediction, target),
            "oracle_source_score": max(scores),
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    source_scores = np.asarray([record["scores"] for record in records], dtype=np.float64)
    route_scores = np.asarray([record["route_score"] for record in records], dtype=np.float64)
    # Pixel-level class-source oracle: choose the best of the three model
    # probabilities independently per class, then use the fixed calibration.
    class_oracle_scores = []
    for (image_path, mask_path), record in zip(samples, records):
        loaded = np.load(cache_dir / f"{image_path.stem}.npz")
        values = [loaded[f"p{i}"].astype(np.float32) for i in range(3)]
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        choices = []
        for source_ids in np.ndindex(*(3,) * 4):
            logits = np.asarray([np.log(np.maximum(values[source_ids[c]][c], 1e-7))
                                 for c in range(4)], dtype=np.float32)
            logits += BIASES[:, None, None]
            prediction = logits.argmax(axis=0).astype(np.uint8)
            choices.append(score_one(prediction, target))
        class_oracle_scores.append(max(choices))

    result = {
        "mean_source": source_scores.mean(axis=0).tolist(),
        "mean_practical_entropy_route": float(route_scores.mean()),
        "mean_image_oracle_source": float(source_scores.max(axis=1).mean()),
        "mean_class_source_oracle": float(np.mean(class_oracle_scores)),
        "records": records,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
