#!/usr/bin/env python3
"""Train a lightweight Al-vs-non-Al refiner on frozen ensemble features."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

from advanced_inference import class_conditional, metrics, predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask, encode_mask


class AlMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def image_features(image_path, probabilities):
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    local_mean = uniform_filter(image, size=15, mode="reflect")
    local_sq = uniform_filter(image * image, size=15, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 1e-6))
    gradient = np.sqrt(sobel(image, axis=0, mode="reflect") ** 2 + sobel(image, axis=1, mode="reflect") ** 2)
    gradient = gradient / max(float(np.percentile(gradient, 99.0)), 1e-6)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-7))).sum(axis=0)
    features = np.concatenate([
        np.log(np.maximum(probabilities, 1e-7)),
        image[None], local_mean[None], local_std[None], gradient[None], entropy[None],
    ], axis=0)
    return features.transpose(1, 2, 0).reshape(-1, features.shape[0]).astype(np.float32)


def load_ensemble_probabilities(args, samples, device, use_amp):
    specs = [
        (args.checkpoint98, 512, 384, 4),
        (args.checkpoint99, 512, 384, 4),
        (args.checkpoint101, 640, 512, 2),
    ]
    models = [load_model(path, device) for path, _, _, _ in specs]
    result = []
    for image_path, mask_path in _local_progress(samples, 'al_refiner'):
        probabilities = []
        for model, (_, patch_size, stride, batch_size) in zip(models, specs):
            probabilities.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                       batch_size, "d4", "gaussian"))
        combined = class_conditional(probabilities)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        result.append((image_path, mask_path, combined, target))
        print(f"processed {image_path.name}", flush=True)
    return result


def sample_training_features(cached, per_image, seed):
    rng = np.random.default_rng(seed)
    all_x, all_y = [], []
    for image_path, _, probabilities, target in cached:
        x = image_features(image_path, probabilities)
        y = (target.reshape(-1) == 3).astype(np.float32)
        positive = np.flatnonzero(y > 0.5)
        negative = np.flatnonzero(y < 0.5)
        n_pos = min(len(positive), per_image // 2)
        n_neg = min(len(negative), per_image - n_pos)
        pos_indices = rng.choice(positive, n_pos, replace=False)
        neg_indices = rng.choice(negative, n_neg, replace=False)
        indices = np.concatenate([pos_indices, neg_indices])
        rng.shuffle(indices)
        all_x.append(x[indices])
        all_y.append(y[indices])
    return np.concatenate(all_x), np.concatenate(all_y)


def train_refiner(x, y, args, device):
    model = AlMLP(x.shape[1]).to(device)
    inputs = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.refiner_lr, weight_decay=1e-4)
    positive_weight = float((y < 0.5).sum() / max((y > 0.5).sum(), 1))
    for epoch in _local_progress(range(args.refiner_epochs), 'al_refiner'):
        model.train()
        logits = model(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(positive_weight, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"refiner epoch={epoch + 1} loss={loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate_refiner(model, cached, args, device):
    model.eval()
    base_bias = np.asarray([0.0, 0.0, 0.10, -0.20], dtype=np.float32)[:, None, None]
    records = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25):
        for bias in (-0.30, -0.20, -0.10, 0.0):
            predictions, targets = [], []
            for image_path, _, probabilities, target in cached:
                x = torch.from_numpy(image_features(image_path, probabilities)).to(device)
                refiner_logit = model(x).cpu().numpy().reshape(target.shape)
                base_al_logit = np.log(np.maximum(probabilities[3], 1e-7)) - np.log(np.maximum(1.0 - probabilities[3], 1e-7))
                adjusted = np.log(np.maximum(probabilities, 1e-7)) + base_bias
                adjusted[3] += float(bias) + float(alpha) * (refiner_logit - base_al_logit)
                predictions.append(adjusted.argmax(axis=0).astype(np.uint8))
                targets.append(target)
            result = metrics(predictions, targets)
            records.append({"alpha": alpha, "al_bias": bias, **result})
    return records


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_cached = load_ensemble_probabilities(args, collect_split(args.data_dir, "train"), device, use_amp)
    valid_cached = load_ensemble_probabilities(args, collect_split(args.data_dir, "valid"), device, use_amp)
    x, y = sample_training_features(train_cached, args.samples_per_image, args.seed)
    print(f"refiner samples={len(y)} features={x.shape[1]} positive={float(y.mean()):.4f}", flush=True)
    refiner = train_refiner(x, y, args, device)
    records = evaluate_refiner(refiner, valid_cached, args, device)
    best = max(records, key=lambda item: item["image_miou_compat"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": refiner.state_dict(), "input_dim": x.shape[1]}, output.with_suffix(".pt"))
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--refiner-epochs", type=int, default=30)
    parser.add_argument("--refiner-lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
