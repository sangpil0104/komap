#!/usr/bin/env python3
"""Global unlabeled-domain TENT adaptation over all validation images."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval(); return model


def corner_batch(image, device):
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), 768)
    h, w = scaled.shape[:2]; ys, xs = tile_starts(h, 768, 576), tile_starts(w, 768, 576)
    locations = list(dict.fromkeys([(ys[0], xs[0]), (ys[0], xs[-1]), (ys[-1], xs[0]), (ys[-1], xs[-1])]))
    return torch.stack([normalize_image(scaled[y:y + 768, x:x + 768]) for y, x in locations]).to(device)


def global_adapt(checkpoint, sources, mode, device, amp, steps=5, lr=1e-5):
    model = load_aspp(checkpoint, device)
    tensors = []
    for item in sources:
        if mode in ("raw", "joint"): tensors.append(corner_batch(item["raw"], device))
        if mode in ("style", "joint"): tensors.append(corner_batch(item["style"], device))
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None: module.weight.requires_grad_(True); params.append(module.weight)
            if module.bias is not None: module.bias.requires_grad_(True); params.append(module.bias)
    model.context.pool[2].eval()
    ids = {id(p) for p in params}
    for p in model.parameters():
        if id(p) not in ids: p.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)
    # Each tensor already contains four 768px corner tiles; keep one image
    # group per optimizer accumulation step on the 16GB P100.
    batch_size = 1
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[start:start + batch_size], dim=0)
            probabilities = torch.softmax(model(batch), dim=1)
            entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-7))).sum(1).mean() / max(1, len(tensors) // batch_size)
            entropy.backward()
        optimizer.step()
    del tensors, optimizer; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    model.eval(); return model


def compose(entries, highs):
    predictions, targets = [], []
    for item, high in zip(entries, highs):
        high_weight = .55 if item["mean"] >= 165.0 else .75; base = MATRIX * (1.0 - high_weight - .15)
        mixed = np.asarray([
            sum(base[c, m] * item["p"][m][c] for m in range(3)) + high_weight * high[c] + .15 * item["pair"][c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (item["gray"] - item["mean"]) / max(item["std"], 1e-5); stats = []
        for cid in range(4):
            w = np.maximum(mixed[cid], 1e-7) ** 2.0
            m = float((w * z).sum() / w.sum()); s = float(np.sqrt((w * (z - m) ** 2).sum() / w.sum()))
            stats.append((m, max(s, .15)))
        adaptive = np.asarray([-0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats], dtype=np.float32)
        prior = item["base_prior"].copy(); prior[0] += 7.0 * (adaptive[0] - prior[0])
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += .40; logits[3] += -.30 if item["mean"] >= 165.0 else -1.0
        predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)); targets.append(item["target"])
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    stats = []
    for p, _ in collect_split(args.data_dir, "train"):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32); stats.append((float(g.mean()), float(g.std())))
    train_mean = float(np.mean([x[0] for x in stats])); train_std = float(np.mean([x[1] for x in stats]))
    pair = load_aspp(args.pair_checkpoint, device); entries = []
    for image_path, mask_path in samples:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8); gray = raw[..., 0].astype(np.float32)
        entries.append({"raw": raw, "style": transform_image(raw, "standardize", train_mean, train_std),
                        "gray": gray, "mean": float(gray.mean()), "std": float(gray.std()),
                        "p": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "pair": predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian"),
                        "base_prior": loaded["prior"].astype(np.float32),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")), "name": image_path.name})
        print(f"cached {image_path.name}", flush=True)
    records = []
    for mode in ("raw", "style", "joint"):
        adapted = global_adapt(args.high_checkpoint, entries, mode, device, amp, 5, 1e-5)
        highs = []
        for item in entries:
            raw_high = predict_array(adapted, item["raw"], 768, 576, device, amp, 1, "d4", "gaussian")
            style_high = predict_array(adapted, item["style"], 768, 576, device, amp, 1, "d4", "gaussian")
            highs.append(.25 * raw_high + .75 * style_high if item["std"] >= 50.0 else raw_high)
        row = {"mode": mode, **compose(entries, highs)}; records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del adapted, highs; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True); main(p.parse_args())
