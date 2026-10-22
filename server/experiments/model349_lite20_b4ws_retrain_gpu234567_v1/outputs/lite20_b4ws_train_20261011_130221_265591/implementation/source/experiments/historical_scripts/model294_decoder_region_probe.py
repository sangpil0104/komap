#!/usr/bin/env python3
"""Frozen decoder-feature probe for component-level phase relabeling."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from advanced_inference import blend_window
from improve_five_causes import collect_split, decode_mask, specimen_id, tile_starts
from model181_raw_zcdf_aspp import representation
from model239_native_stem_finetune import Stride1NativeStemASPP
from model253_highres_scale_finetune import resize_pair
from model288_region_classifier_cv import extract_regions, image_score


class RegionFeatureASPP(Stride1NativeStemASPP):
    def forward_features(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))
        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        decoded = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        joined = torch.cat([decoded, stem_skip], dim=1)
        head_block = self.head[0].block
        hidden = head_block[0](joined) + self.native_to_head(self.native_stem(x))
        hidden = head_block[2](head_block[1](hidden))
        hidden = head_block[5](head_block[4](head_block[3](hidden)))
        return self.head[1](hidden), hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]


class RegionMLP3(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 3),
        )

    def forward(self, x):
        return self.layers(x)


def _infer_maps(model, image_path: Path, patch_size: int, stride: int,
                device: torch.device, amp: bool):
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = resize_pair(source, np.zeros(source.shape[:2], dtype=np.uint8),
                            scale=1.0, minimum_side=patch_size)
    features = representation(scaled)
    height, width = features.shape[-2:]
    ys = tile_starts(height, patch_size, stride)
    xs = tile_starts(width, patch_size, stride)
    padded_height = max(height, ys[-1] + patch_size)
    padded_width = max(width, xs[-1] + patch_size)
    padded = np.pad(features, ((0, 0), (0, padded_height - height),
                               (0, padded_width - width)), mode="reflect")
    window = blend_window(patch_size, "gaussian")
    logits_sum = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    hidden_sum = np.zeros((32, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    with torch.no_grad():
        for top in ys:
            for left in xs:
                inputs = torch.from_numpy(padded[:, top:top + patch_size,
                                                 left:left + patch_size][None]).to(
                                                     device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits, hidden = model.forward_features(inputs)
                    probabilities = torch.softmax(logits.float(), dim=1)
                logits_np = probabilities[0].float().cpu().numpy()
                hidden_np = hidden[0].float().cpu().numpy()
                logits_sum[:, top:top + patch_size, left:left + patch_size] += (
                    logits_np * window[None])
                hidden_sum[:, top:top + patch_size, left:left + patch_size] += (
                    hidden_np * window[None])
                weight_sum[top:top + patch_size, left:left + patch_size] += window
    probabilities = logits_sum[:, :height, :width] / np.maximum(
        weight_sum[None, :height, :width], 1e-7)
    probabilities = np.stack([
        np.asarray(Image.fromarray(probabilities[c]).resize(
            (source.shape[1], source.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
        for c in range(4)
    ])
    probabilities /= np.maximum(probabilities.sum(0, keepdims=True), 1e-7)
    hidden_map = hidden_sum[:, :height, :width] / np.maximum(
        weight_sum[None, :height, :width], 1e-7)
    hidden_map = np.stack([
        np.asarray(Image.fromarray(value).resize(
            (source.shape[1], source.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
        for value in hidden_map
    ])
    return source, probabilities, hidden_map


def _pool_hidden(hidden: np.ndarray, component_map: np.ndarray,
                 component_ids: np.ndarray) -> np.ndarray:
    count = int(component_ids.max(initial=0))
    if count == 0:
        return np.empty((0, 64), dtype=np.float32)
    ids = component_map.reshape(-1).astype(np.int64)
    keep = ids > 0
    ids = ids[keep]
    area = np.bincount(ids, minlength=count + 1)[1:].astype(np.float64)
    pieces = []
    for channel in hidden:
        values = channel.reshape(-1)[keep].astype(np.float64)
        sums = np.bincount(ids, weights=values, minlength=count + 1)[1:]
        squares = np.bincount(ids, weights=values * values, minlength=count + 1)[1:]
        mean = sums / np.maximum(area, 1.0)
        std = np.sqrt(np.maximum(squares / np.maximum(area, 1.0) - mean * mean, 0.0))
        pieces.extend((mean, std))
    return np.stack(pieces, axis=1).astype(np.float32)


def _record(image_path: Path, mask_path: Path, probabilities: np.ndarray,
            hidden: np.ndarray, train_gt_components: bool):
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = target.copy() if train_gt_components else probabilities.argmax(0).astype(np.uint8)
    represented = representation(source)
    features, labels, weights, predicted, component_ids, component_map = extract_regions(
        prediction, target, probabilities, {"z": represented[1], "q": represented[2]})
    learned = _pool_hidden(hidden, component_map, component_ids)
    if len(features):
        features = np.concatenate((features, learned), axis=1)
    else:
        features = np.empty((0, 96), dtype=np.float32)
    return {
        "name": image_path.stem,
        "group": specimen_id(image_path),
        "target": target,
        "prediction": probabilities.argmax(0).astype(np.uint8),
        "base_probs": probabilities.astype(np.float32),
        "base_score": image_score(probabilities.argmax(0), target),
        "features": features,
        "labels": labels,
        "weights": weights,
        "predicted": predicted,
        "component_ids": component_ids,
        "component_map": component_map,
    }


def _fit_predict(train_x, train_y, train_w, test_x, args):
    mean = train_x.mean(0, keepdims=True)
    std = np.maximum(train_x.std(0, keepdims=True), 1e-4)
    train_x = np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32)
    test_x = np.clip((test_x - mean) / std, -10.0, 10.0).astype(np.float32)
    y_map = {0: 0, 1: 1, 3: 2}
    train_y = np.asarray([y_map[int(value)] for value in train_y], dtype=np.int64)
    weight = train_w / max(float(train_w.mean()), 1e-8)
    x_t = torch.from_numpy(train_x)
    y_t = torch.from_numpy(train_y)
    w_t = torch.from_numpy(weight.astype(np.float32))
    test_t = torch.from_numpy(test_x)
    torch.manual_seed(args.seed)
    model = RegionMLP3(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batch_size = 2048
    model.train()
    for _ in _local_progress(range(args.epochs), 'model294_decoder_region_probe'):
        order = torch.randperm(len(x_t))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = F.cross_entropy(model(x_t[indices]), y_t[indices], reduction="none")
            loss = (losses * w_t[indices]).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(test_t), dim=1).numpy()


def _apply(record, region_probs, threshold, soft_blend):
    output = record["prediction"].copy()
    soft = record["base_probs"].copy()
    class_ids = np.asarray((0, 1, 3), dtype=np.uint8)
    labels = class_ids[region_probs.argmax(1)] if len(region_probs) else np.empty(0, np.uint8)
    confidence = region_probs.max(1) if len(region_probs) else np.empty(0, np.float32)
    for index, label in enumerate(labels):
        if confidence[index] < threshold or int(label) == int(record["predicted"][index]):
            continue
        region = record["component_map"] == record["component_ids"][index]
        output[region] = label
        region_probability = np.zeros(4, dtype=np.float32)
        region_probability[class_ids] = region_probs[index]
        soft[:, region] = ((1.0 - soft_blend) * soft[:, region]
                           + soft_blend * region_probability[:, None])
    soft /= np.maximum(soft.sum(0, keepdims=True), 1e-7)
    return output, soft


def main(args):
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model = RegionFeatureASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    valid_archive = np.load(args.valid_probs)
    train_parts, label_parts, weight_parts = [], [], []
    start_time = time.time()

    for index, (image_path, mask_path) in enumerate(train_samples, 1):
        _, probabilities, hidden = _infer_maps(
            model, image_path, args.patch_size, args.val_stride, device, amp)
        record = _record(image_path, mask_path, probabilities, hidden, True)
        if len(record["labels"]):
            train_parts.append(record["features"])
            label_parts.append(record["labels"])
            weight_parts.append(record["weights"])
        if index % 10 == 0 or index == len(train_samples):
            print(f"train pooled features {index}/{len(train_samples)} "
                  f"regions={sum(len(x) for x in label_parts)} "
                  f"elapsed={time.time() - start_time:.1f}s", flush=True)

    train_x = np.concatenate(train_parts).astype(np.float32)
    train_y = np.concatenate(label_parts).astype(np.int64)
    train_w = np.concatenate(weight_parts).astype(np.float32)
    records = []
    for index, (image_path, mask_path) in enumerate(_local_progress(valid_samples, 'model294_decoder_region_probe'), 1):
        source, _, hidden = _infer_maps(
            model, image_path, args.patch_size, args.val_stride, device, amp)
        probabilities = valid_archive[image_path.stem].astype(np.float32)
        record = _record(image_path, mask_path, probabilities, hidden, False)
        records.append(record)
        if index % 5 == 0 or index == len(valid_samples):
            print(f"valid pooled features {index}/{len(valid_samples)} "
                  f"elapsed={time.time() - start_time:.1f}s", flush=True)

    all_valid_x = np.concatenate([record["features"] for record in records])
    region_probs = _fit_predict(train_x, train_y, train_w, all_valid_x, args)
    cursor = 0
    counts = []
    for record in records:
        count = len(record["labels"])
        record["region_probs"] = region_probs[cursor:cursor + count]
        cursor += count

    thresholds = tuple(args.thresholds)
    scores = {value: np.zeros(len(records), dtype=np.float64) for value in thresholds}
    fixed = float(args.fixed_threshold)
    fixed = min(thresholds, key=lambda value: abs(value - fixed))
    fixed_soft = []
    baseline = np.asarray([record["base_score"] for record in records])
    for index, record in enumerate(records):
        for threshold in thresholds:
            prediction, soft = _apply(record, record["region_probs"], threshold,
                                      args.soft_blend)
            scores[threshold][index] = image_score(prediction, record["target"])
            if threshold == fixed:
                fixed_soft.append(soft)

    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["group"], []).append(index)
    cv = np.zeros(len(records), dtype=np.float64)
    selected_by_group = {}
    for group, heldout in groups.items():
        other = [index for name, ids in groups.items() if name != group for index in ids]
        selected = max(thresholds, key=lambda t: float(scores[t][other].mean()))
        cv[heldout] = scores[selected][heldout]
        selected_by_group[group] = float(selected)
    delta = cv - baseline
    rng = np.random.default_rng(args.seed)
    group_names = list(groups)
    bootstrap = np.empty(args.bootstrap_draws, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        ids = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[ids].mean() * 100.0
    train_groups = {specimen_id(path) for path, _ in train_samples}
    overlap = np.asarray([record["group"] in train_groups for record in records])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "valid_probs_decoder_region.npz", **{
        record["name"]: value.astype(np.float16)
        for record, value in zip(records, fixed_soft)
    })
    result = {
        "method": "frozen decoder hidden feature + mask-pooled region MLP probe",
        "train_image_count": len(train_samples),
        "train_component_count": int(len(train_y)),
        "valid_image_count": len(records),
        "feature_count": int(train_x.shape[1]),
        "baseline_image_mean_miou": float(baseline.mean()),
        "thresholds": {str(t): {
            "image_mean_miou": float(scores[t].mean()),
            "delta_pp_vs_baseline": float((scores[t] - baseline).mean() * 100.0),
            "positive_image_count": int((scores[t] > baseline).sum()),
        } for t in thresholds},
        "group_cv": {
            "image_mean_miou": float(cv.mean()),
            "delta_pp_vs_baseline": float(delta.mean() * 100.0),
            "positive_image_count": int((delta > 0).sum()),
            "group_bootstrap_95ci_delta_pp": [
                float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "threshold_by_heldout_group": selected_by_group,
            "nonoverlap_image_count": int((~overlap).sum()),
            "nonoverlap_delta_pp": float(delta[~overlap].mean() * 100.0),
            "overlap_image_count": int(overlap.sum()),
            "overlap_delta_pp": float(delta[overlap].mean() * 100.0),
        },
        "fixed_threshold": fixed,
        "elapsed_seconds": time.time() - start_time,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(0.75, 0.85, 0.90, 0.95))
    parser.add_argument("--fixed-threshold", type=float, default=0.90)
    parser.add_argument("--soft-blend", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=294)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
