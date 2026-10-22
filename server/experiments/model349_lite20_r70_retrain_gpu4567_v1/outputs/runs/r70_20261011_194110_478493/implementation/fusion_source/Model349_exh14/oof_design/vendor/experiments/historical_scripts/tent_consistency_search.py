#!/usr/bin/env python3
"""TENT with raw/style prediction consistency for domain adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def adapt_consistent(checkpoint, raw, train_mean, train_std, device, use_amp, steps, lr, consistency_weight):
    model = load_aspp(checkpoint, device)
    style = transform_image(raw, "standardize", train_mean, train_std)
    raw_scaled, _ = scaled_pair(raw, np.zeros(raw.shape[:2], dtype=np.int64), 768)
    style_scaled, _ = scaled_pair(style, np.zeros(style.shape[:2], dtype=np.int64), 768)
    height, width = raw_scaled.shape[:2]
    ys, xs = tile_starts(height, 768, 576), tile_starts(width, 768, 576)
    locations = list(dict.fromkeys((y, x) for y in ys for x in xs))
    raw_inputs = torch.stack([normalize_image(raw_scaled[y:y + 768, x:x + 768]) for y, x in locations]).to(device)
    style_inputs = torch.stack([normalize_image(style_scaled[y:y + 768, x:x + 768]) for y, x in locations]).to(device)
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True); params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True); params.append(module.bias)
    model.context.pool[2].eval()
    param_ids = {id(p) for p in params}
    for parameter in model.parameters():
        if id(parameter) not in param_ids:
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # A pair of 768px tiles is too large for the P100 when retained as one
        # graph. Backpropagate each raw/style tile pair before moving on; BN
        # affine gradients accumulate until the optimizer step.
        for index in range(raw_inputs.shape[0]):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                raw_prob = torch.softmax(model(raw_inputs[index:index + 1]), dim=1)
                style_prob = torch.softmax(model(style_inputs[index:index + 1]), dim=1)
                raw_entropy = -(raw_prob * torch.log(torch.clamp(raw_prob, min=1e-7))).sum(1).mean()
                style_entropy = -(style_prob * torch.log(torch.clamp(style_prob, min=1e-7))).sum(1).mean()
                raw_log = torch.log(torch.clamp(raw_prob, min=1e-7))
                style_log = torch.log(torch.clamp(style_prob, min=1e-7))
                consistency = 0.5 * (
                    F.kl_div(raw_log, style_prob.detach(), reduction="batchmean")
                    + F.kl_div(style_log, raw_prob.detach(), reduction="batchmean")
                )
                loss = (0.5 * (raw_entropy + style_entropy)
                        + consistency_weight * consistency) / raw_inputs.shape[0]
            loss.backward()
        optimizer.step()
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    entries = []
    for image_path, mask_path in samples:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({"values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "prior": loaded["prior"].astype(np.float32), "raw": raw,
                        "std": float(raw[..., 0].astype(np.float32).std()),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")),
                        "name": image_path.name})
    pair = load_aspp(args.pair_checkpoint, device)
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    records = []
    for consistency_weight in (0.0, 0.2, 0.5, 1.0):
        predictions, targets = [], []
        for entry, (image_path, _) in zip(entries, samples):
            pair_value = predict_probabilities(pair, image_path, 768, 576, device,
                                                use_amp, 1, "d4", "gaussian")
            model = adapt_consistent(args.high_checkpoint, entry["raw"], train_mean, train_std,
                                     device, use_amp, args.steps, args.lr, consistency_weight)
            raw_high = predict_array(model, entry["raw"], 768, 576, device,
                                     use_amp, 1, "d4", "gaussian")
            style = transform_image(entry["raw"], "standardize", train_mean, train_std)
            style_high = predict_array(model, style, 768, 576, device,
                                       use_amp, 1, "d4", "gaussian")
            high = (0.25 * raw_high + 0.75 * style_high
                    if entry["std"] >= args.route_threshold else raw_high)
            base = MATRIX * (0.40 - args.pair_weight)
            mixed = np.asarray([
                sum(base[c, m] * entry["values"][m][c] for m in range(3))
                + 0.60 * high[c] + args.pair_weight * pair_value[c]
                for c in range(4)
            ], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
            logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"]
            logits[2] += 0.40; logits[3] -= 0.50
            predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
            targets.append(entry["target"])
            print(f"weight={consistency_weight} adapted {entry['name']}", flush=True)
        value = metrics(predictions, targets)
        row = {"consistency_weight": consistency_weight, "steps": args.steps,
               "lr": args.lr, "pair_weight": args.pair_weight, **value}
        records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--pair-weight", type=float, default=.10)
    p.add_argument("--route-threshold", type=float, default=50.0); p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
