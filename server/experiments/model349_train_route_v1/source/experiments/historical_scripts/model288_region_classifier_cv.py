#!/usr/bin/env python3
"""Leave-one-specimen-out region relabeling probe for the model225 route."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict


def image_score(prediction: np.ndarray, target: np.ndarray) -> float:
    matrix = np.bincount(
        target.reshape(-1).astype(np.int64) * 4 + prediction.reshape(-1).astype(np.int64),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return float((intersection / np.maximum(union, 1.0)).mean())


def _region_moments(labels: np.ndarray, binary: np.ndarray, values: np.ndarray,
                    count: int) -> tuple[np.ndarray, np.ndarray]:
    ids = labels[binary]
    vals = values[binary].astype(np.float64)
    sums = np.bincount(ids, weights=vals, minlength=count + 1)[1:]
    squares = np.bincount(ids, weights=vals * vals, minlength=count + 1)[1:]
    area = np.bincount(ids, minlength=count + 1)[1:].astype(np.float64)
    mean = sums / np.maximum(area, 1.0)
    variance = np.maximum(squares / np.maximum(area, 1.0) - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def extract_regions(prediction: np.ndarray, target: np.ndarray, mixed: np.ndarray,
                    item: dict, connectivity: int = 1) -> tuple[np.ndarray, ...]:
    height, width = target.shape
    z = np.asarray(item["z"], dtype=np.float32)
    q = np.asarray(item["q"], dtype=np.float32)
    z_local = ndimage.gaussian_filter(z, sigma=5.0, mode="reflect")
    z_local_std = np.sqrt(np.maximum(
        ndimage.gaussian_filter(z * z, sigma=5.0, mode="reflect") - z_local * z_local, 0.0))
    q_local = ndimage.gaussian_filter(q, sigma=5.0, mode="reflect")
    q_local_std = np.sqrt(np.maximum(
        ndimage.gaussian_filter(q * q, sigma=5.0, mode="reflect") - q_local * q_local, 0.0))
    grad_z = np.hypot(*np.gradient(z)).astype(np.float32)
    structure = ndimage.generate_binary_structure(2, connectivity)

    feature_parts, labels_out, weights_out = [], [], []
    predicted_parts, component_id_parts = [], []
    component_map = np.zeros(target.shape, dtype=np.int32)
    component_offset = 0
    for class_id in (0, 1, 3):
        binary = prediction == class_id
        labels, count = ndimage.label(binary, structure=structure)
        if count == 0:
            continue
        component_map[binary] = labels[binary] + component_offset
        ids = labels[binary]
        area = np.bincount(ids, minlength=count + 1)[1:].astype(np.float64)
        encoded = ids.astype(np.int64) * 4 + target[binary].astype(np.int64)
        class_counts = np.bincount(encoded, minlength=(count + 1) * 4).reshape(
            count + 1, 4)
        majority = class_counts.argmax(axis=1)[1:].astype(np.int64)

        features = [np.eye(3, dtype=np.float32)[(class_id, 1, 3).index(class_id)]
                    .reshape(1, -1).repeat(count, axis=0)]
        probability_means, probability_stds = [], []
        for channel in range(4):
            mean, std = _region_moments(labels, binary, mixed[channel], count)
            probability_means.append(mean)
            probability_stds.append(std)
        mean_probability = np.stack(probability_means, axis=1)
        std_probability = np.stack(probability_stds, axis=1)
        entropy = -(mean_probability * np.log(np.maximum(mean_probability, 1e-7))).sum(1)
        ordered = np.sort(mean_probability, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        features.extend((mean_probability, std_probability, entropy[:, None], margin[:, None],
                         np.log1p(area)[:, None]))

        yy, xx = np.nonzero(binary)
        ids_float = ids.astype(np.int64)
        x = (xx.astype(np.float32) / max(width - 1, 1)) * 2.0 - 1.0
        y = (yy.astype(np.float32) / max(height - 1, 1)) * 2.0 - 1.0

        def moments(values: np.ndarray) -> np.ndarray:
            return (np.bincount(ids_float, weights=values.astype(np.float64),
                                minlength=count + 1)[1:] / np.maximum(area, 1.0))

        mx, my = moments(x), moments(y)
        vx = np.maximum(moments(x * x) - mx * mx, 0.0)
        vy = np.maximum(moments(y * y) - my * my, 0.0)
        vxy = moments(x * y) - mx * my
        trace = vx + vy
        eig_delta = np.sqrt(np.maximum((vx - vy) ** 2 + 4.0 * vxy * vxy, 0.0))
        eig_max = np.maximum((trace + eig_delta) * 0.5, 1e-10)
        eig_min = np.maximum((trace - eig_delta) * 0.5, 0.0)
        elongation = np.sqrt(eig_min / eig_max)
        boundary = binary & ~ndimage.binary_erosion(binary, structure=structure,
                                                     border_value=0)
        perimeter = np.bincount(labels[boundary], minlength=count + 1)[1:].astype(np.float64)
        compactness = 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0)
        features.extend((mx[:, None], my[:, None], elongation[:, None].astype(np.float32),
                         np.log1p(compactness)[:, None].astype(np.float32)))

        for values in (z, q, z_local, z_local_std, q_local, q_local_std, grad_z):
            mean, std = _region_moments(labels, binary, values, count)
            features.extend((mean[:, None], std[:, None]))

        feature_parts.append(np.concatenate(features, axis=1).astype(np.float32))
        labels_out.append(majority)
        gt_area = np.bincount(target.reshape(-1), minlength=4).astype(np.float64)
        weights_out.append(area / np.maximum(gt_area[majority], 1.0))
        predicted_parts.append(np.full(count, class_id, dtype=np.int64))
        component_id_parts.append(np.arange(component_offset + 1,
                                            component_offset + count + 1,
                                            dtype=np.int64))
        component_offset += count

    if not feature_parts:
        return (np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64), component_map)
    features = np.concatenate(feature_parts)
    labels = np.concatenate(labels_out)
    sample_weights = np.concatenate(weights_out).astype(np.float32)
    predicted_classes = np.concatenate(predicted_parts)
    component_ids = np.concatenate(component_id_parts)
    return features, labels, sample_weights, predicted_classes, component_ids, component_map


class RegionMLP(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, train_w: np.ndarray,
                test_x: np.ndarray, seed: int, epochs: int = 25) -> np.ndarray:
    torch.manual_seed(seed)
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-4)
    train_x = np.clip((train_x - mean) / std, -10.0, 10.0)
    test_x = np.clip((test_x - mean) / std, -10.0, 10.0)
    train_x_t = torch.from_numpy(train_x.astype(np.float32))
    train_y_t = torch.from_numpy(train_y.astype(np.int64))
    train_w = train_w / max(float(train_w.mean()), 1e-8)
    train_w_t = torch.from_numpy(train_w.astype(np.float32))
    test_x_t = torch.from_numpy(test_x.astype(np.float32))
    model = RegionMLP(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batch_size = 2048
    model.train()
    for _ in _local_progress(range(epochs), 'model288_region_classifier_cv'):
        order = torch.randperm(train_x_t.shape[0])
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x_t[indices])
            losses = F.cross_entropy(logits, train_y_t[indices], reduction="none")
            loss = (losses * train_w_t[indices]).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(test_x_t), dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    torch.set_num_threads(4)

    with Path(args.entries).open("rb") as stream:
        entries = pickle.load(stream)
    with Path(args.settings).open(encoding="utf-8") as stream:
        setting = json.load(stream)["best_image"]
    archives = {
        "195": np.load(args.model195), "217": np.load(args.tent20),
        "244": np.load(args.model244), "249": np.load(args.model249),
    }
    key_orders = {tuple(archive.files) for archive in archives.values()}
    if len(key_orders) != 1 or len(next(iter(key_orders))) != len(entries):
        raise ValueError("validation probability archives do not align with cache entries")
    keys = tuple(next(iter(key_orders)))
    values = {source: [archive[key].astype(np.float32) for key in keys]
              for source, archive in archives.items()}

    records = []
    for index, (name, item) in enumerate(zip(keys, entries)):
        row = {
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{source: archive_values[index]
                           for source, archive_values in values.items()}},
        }
        mixed = make_mixed(row, setting)
        prediction = predict(item, mixed, setting, bilateral_weights(item["z"]), exact=True)
        features, labels, weights, predicted, component_ids, component_map = extract_regions(
            prediction, item["target"].astype(np.uint8), mixed, item)
        records.append({
            "name": name, "group": name.split("_", 1)[0],
            "prediction": prediction.astype(np.uint8),
            "target": item["target"].astype(np.uint8),
            "features": features, "labels": labels, "weights": weights,
            "predicted": predicted, "component_ids": component_ids,
            "component_map": component_map,
            "base_score": image_score(prediction, item["target"]),
        })
        print(f"features {name}: {len(labels)} components, {features.shape[1]} features",
              flush=True)

    groups = sorted({record["group"] for record in records})
    thresholds = (0.0, 0.60, 0.75, 0.90)
    corrected_scores = {threshold: np.zeros(len(records), dtype=np.float64)
                        for threshold in thresholds}
    baseline_scores = np.asarray([record["base_score"] for record in records])
    for fold_index, heldout_group in enumerate(groups):
        heldout = [i for i, record in enumerate(records) if record["group"] == heldout_group]
        training = [i for i, record in enumerate(records) if record["group"] != heldout_group]
        train_x = np.concatenate([records[i]["features"] for i in training])
        train_y = np.concatenate([records[i]["labels"] for i in training])
        train_w = np.concatenate([records[i]["weights"] for i in training])
        test_x = np.concatenate([records[i]["features"] for i in heldout])
        probabilities = fit_predict(train_x, train_y, train_w, test_x,
                                    seed=288 + fold_index, epochs=args.epochs)
        cursor = 0
        for image_index in heldout:
            record = records[image_index]
            count = len(record["labels"])
            prob = probabilities[cursor:cursor + count]
            cursor += count
            predicted_region = prob.argmax(axis=1)
            confidence = prob.max(axis=1)
            for threshold in thresholds:
                output = record["prediction"].copy()
                region_labels = np.where(confidence >= threshold,
                                         predicted_region, record["predicted"])
                for index, new_label in enumerate(region_labels):
                    if new_label != record["predicted"][index]:
                        output[record["component_map"] == record["component_ids"][index]] = new_label
                corrected_scores[threshold][image_index] = image_score(
                    output, record["target"])
        print(f"completed held-out specimen {heldout_group} ({fold_index + 1}/{len(groups)})",
              flush=True)

    group_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        group_indices.setdefault(record["group"], []).append(index)
    rng = np.random.default_rng(288)
    result = {
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "method": "class-specific 4-connected component MLP relabeling; specimen LOSO",
        "region_target": "ground-truth majority class within predicted component",
        "feature_count": int(records[0]["features"].shape[1]),
        "component_count": int(sum(len(record["labels"]) for record in records)),
        "image_count": len(records), "specimen_group_count": len(groups),
        "baseline_image_mean_miou": float(baseline_scores.mean()),
        "thresholds": {},
    }
    for threshold in thresholds:
        scores = corrected_scores[threshold]
        delta = scores - baseline_scores
        bootstrap = np.empty(10000, dtype=np.float64)
        for draw in range(len(bootstrap)):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            indices = [index for group in sampled for index in group_indices[str(group)]]
            bootstrap[draw] = delta[indices].mean() * 100.0
        result["thresholds"][str(threshold)] = {
            "image_mean_miou": float(scores.mean()),
            "delta_pp_vs_baseline": float(delta.mean() * 100.0),
            "positive_image_count": int((delta > 0).sum()),
            "group_bootstrap_95ci_delta_pp": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "image_scores": scores.tolist(),
        }
    result["images"] = [{
        "name": record["name"], "group": record["group"],
        "baseline": float(baseline_scores[index]),
        **{f"score_threshold_{threshold}": float(corrected_scores[threshold][index])
           for threshold in thresholds},
    } for index, record in enumerate(records)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["thresholds"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
