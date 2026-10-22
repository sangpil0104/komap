#!/usr/bin/env python3
"""Add the corrected ASPP checkpoint to the current best restart ensemble.

The current best protocol is intentionally reproduced here:
  * restart08: native 640px U-Net++/ResNet34
  * restart19: native 512px Al3Ni-weighted hard-negative fine-tune
  * legacy four-view TTA and uniform overlap averaging
  * train-only grayscale class prior

ASPP is evaluated as a third probability member.  All search choices are
made on valid only; test masks are generated only with the selected setting.
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
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask, encode_mask
from restart_ensemble import metrics


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))


def load_aspp(checkpoint: str, base_checkpoint: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=base_checkpoint).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def fit_stats(data_dir: str):
    values = [[] for _ in range(4)]
    for image_path, mask_path in collect_split(Path(data_dir), "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for class_id, color in enumerate(COLORS):
            ids[np.all(mask == color, axis=2)] = class_id
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for class_id in range(4):
            current = z[ids == class_id]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[class_id].append(current[take])
    return [(float(np.concatenate(current).mean()),
             max(float(np.concatenate(current).std()), 0.15)) for current in values]


def combine(old_p, new_p, aspp_p, new_weight, aspp_weight):
    old_weight = 1.0 - new_weight - aspp_weight
    result = old_weight * old_p + new_weight * new_p + aspp_weight * aspp_p
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old_model = load_model(args.old_checkpoint, device)
    new_model = load_model(args.new_checkpoint, device)
    aspp_model = load_aspp(args.aspp_checkpoint, args.aspp_base_checkpoint, device)
    stats = fit_stats(args.data_dir)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        old_p = predict_probabilities(old_model, image_path, 640, 512, device, use_amp, 2, True)
        new_p = predict_probabilities(new_model, image_path, 512, 384, device, use_amp, 2, True)
        aspp_p = predict_probabilities(aspp_model, image_path, 512, 384, device, use_amp, 2, True)
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([
            -0.5 * ((z - mean) / std) ** 2 - np.log(std)
            for mean, std in stats
        ], dtype=np.float32)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append((old_p, new_p, aspp_p, prior, target))
        print(f"cached {image_path.name}", flush=True)

    records = []

    def evaluate_setting(new_weight, aspp_weight, alpha, eutectic_scale, al_bias, eutectic_bias):
        predictions, targets = [], []
        bias = np.asarray([0.0, 0.0, al_bias, eutectic_bias], dtype=np.float32)[:, None, None]
        for old_p, new_p, aspp_p, prior, target in cached:
            probability = combine(old_p, new_p, aspp_p, new_weight, aspp_weight)
            logits = np.log(np.maximum(probability, 1e-7)) + alpha * prior
            logits[3] += alpha * (eutectic_scale - 1.0) * prior[3]
            predictions.append((logits + bias).argmax(axis=0).astype(np.uint8))
            targets.append(target)
        result = metrics(predictions, targets)
        return {"new_weight": new_weight, "aspp_weight": aspp_weight,
                "old_weight": 1.0 - new_weight - aspp_weight, "alpha": alpha,
                "eutectic_scale": eutectic_scale, "al_bias": al_bias,
                "eutectic_bias": eutectic_bias, **result}

    # Stage 1: only the model mixture changes.  This quickly answers whether
    # ASPP supplies complementary information at all.
    coarse = []
    for aspp_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        for new_weight in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            if new_weight + aspp_weight <= 0.95:
                record = evaluate_setting(new_weight, aspp_weight, 0.07, 1.0, 0.10, -0.50)
                records.append(record)
                coarse.append(record)
    coarse_best = max(coarse, key=lambda item: item["image_mean_miou"])

    # Stage 2a: refine only the mixture around the best coarse point.
    for aspp_weight in sorted(set(round(max(0.0, coarse_best["aspp_weight"] + delta), 3)
                                 for delta in (-0.05, 0.0, 0.05))):
        for new_weight in sorted(set(round(max(0.35, coarse_best["new_weight"] + delta), 3)
                                    for delta in (-0.05, 0.0, 0.05))):
            if new_weight + aspp_weight > 0.95:
                continue
            records.append(evaluate_setting(new_weight, aspp_weight, 0.07, 1.0, 0.10, -0.50))

    mixture_best = max(records, key=lambda item: item["image_mean_miou"])
    # Stage 2b: tune calibration at the selected mixture only.
    for alpha in (0.05, 0.07, 0.09):
        for eutectic_scale in (0.75, 1.0, 1.25):
            for al_bias in (0.0, 0.10, 0.20):
                for eutectic_bias in (-0.55, -0.50, -0.45):
                    records.append(evaluate_setting(mixture_best["new_weight"],
                                                     mixture_best["aspp_weight"], alpha,
                                                     eutectic_scale, al_bias, eutectic_bias))
    best = max(records, key=lambda item: item["image_mean_miou"])

    # A model can be complementary only for one phase.  Greedily test a
    # conservative class-conditional reweighting around the best uniform
    # mixture; this adds very few candidates and avoids a high-dimensional
    # validation overfit.
    base_weights = np.asarray([[best["old_weight"], best["new_weight"], best["aspp_weight"]]
                               for _ in range(4)], dtype=np.float32)
    class_options = {
        "base": np.asarray([best["old_weight"], best["new_weight"], best["aspp_weight"]], dtype=np.float32),
        "old_heavy": np.asarray([best["old_weight"] + 0.15, best["new_weight"] - 0.10,
                                  best["aspp_weight"] - 0.05], dtype=np.float32),
        "new_heavy": np.asarray([best["old_weight"] - 0.10, best["new_weight"] + 0.15,
                                  best["aspp_weight"] - 0.05], dtype=np.float32),
        "aspp_heavy": np.asarray([best["old_weight"] - 0.10, best["new_weight"] - 0.05,
                                   best["aspp_weight"] + 0.15], dtype=np.float32),
    }
    class_options = {name: value / value.sum() for name, value in class_options.items()}

    def evaluate_class_weights(matrix):
        predictions, targets = [], []
        bias = np.asarray([0.0, 0.0, best["al_bias"], best["eutectic_bias"]], dtype=np.float32)[:, None, None]
        for old_p, new_p, aspp_p, prior, target in cached:
            probability = np.asarray([
                matrix[class_id, 0] * old_p[class_id]
                + matrix[class_id, 1] * new_p[class_id]
                + matrix[class_id, 2] * aspp_p[class_id]
                for class_id in range(4)
            ], dtype=np.float32)
            probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-7)
            logits = np.log(np.maximum(probability, 1e-7)) + best["alpha"] * prior
            logits[3] += best["alpha"] * (best["eutectic_scale"] - 1.0) * prior[3]
            predictions.append((logits + bias).argmax(axis=0).astype(np.uint8))
            targets.append(target)
        return metrics(predictions, targets)

    for class_id in range(4):
        candidates = []
        for option_name, option in class_options.items():
            candidate_matrix = base_weights.copy()
            candidate_matrix[class_id] = option
            metric = evaluate_class_weights(candidate_matrix)
            candidates.append((metric["image_mean_miou"], option_name, candidate_matrix, metric))
        _, option_name, candidate_matrix, metric = max(candidates, key=lambda item: item[0])
        if metric["image_mean_miou"] > best["image_mean_miou"]:
            base_weights = candidate_matrix
            best = {**best, **metric, "class_weights": base_weights.tolist(),
                    "class_update": {"class_id": class_id, "option": option_name}}
    best.setdefault("class_weights", base_weights.tolist())
    result = {"best": best, "records": records, "train_z_stats": stats}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bias = np.asarray([0.0, 0.0, best["al_bias"], best["eutectic_bias"]], dtype=np.float32)[:, None, None]
        for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
            old_p = predict_probabilities(old_model, image_path, 640, 512, device, use_amp, 2, True)
            new_p = predict_probabilities(new_model, image_path, 512, 384, device, use_amp, 2, True)
            aspp_p = predict_probabilities(aspp_model, image_path, 512, 384, device, use_amp, 2, True)
            matrix = np.asarray(best["class_weights"], dtype=np.float32)
            probability = np.asarray([
                matrix[class_id, 0] * old_p[class_id]
                + matrix[class_id, 1] * new_p[class_id]
                + matrix[class_id, 2] * aspp_p[class_id]
                for class_id in range(4)
            ], dtype=np.float32)
            probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-7)
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
            prior = np.asarray([
                -0.5 * ((z - mean) / std) ** 2 - np.log(std)
                for mean, std in stats
            ], dtype=np.float32)
            logits = np.log(np.maximum(probability, 1e-7)) + best["alpha"] * prior
            logits[3] += best["alpha"] * (best["eutectic_scale"] - 1.0) * prior[3]
            prediction = (logits + bias).argmax(axis=0).astype(np.uint8)
            encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote test_masks={len(list(output_dir.glob('*.png')))} dir={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
