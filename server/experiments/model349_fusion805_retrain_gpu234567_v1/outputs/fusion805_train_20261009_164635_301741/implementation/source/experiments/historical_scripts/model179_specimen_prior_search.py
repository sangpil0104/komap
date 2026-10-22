#!/usr/bin/env python3
"""Train-only specimen-aware intensity prior on the model-173 route.

The validation mask is never used to fit the prior.  For a validation image,
train images with the same specimen A-ID provide a local class-intensity
distribution; unseen specimen IDs fall back to the global train distribution.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from highres1024_search import choose_source, make_logits, predict
from improve_five_causes import CLASS_COLORS, collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def specimen_id(path: Path) -> str:
    match = re.match(r"(A\d+)_", path.name)
    if not match:
        raise ValueError(path.name)
    return match.group(1)


def ids_from_mask(mask_rgb: np.ndarray) -> np.ndarray:
    ids = np.full(mask_rgb.shape[:2], -1, dtype=np.int8)
    for class_id, color in enumerate(CLASS_COLORS):
        ids[np.all(mask_rgb == color, axis=2)] = class_id
    return ids


def fit_group_stats(data_dir: Path):
    """Return train-only z/q Gaussian stats by specimen ID."""
    groups = defaultdict(lambda: {"z": [[] for _ in range(4)], "q": [[] for _ in range(4)]})
    for image_path, mask_path in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        q = percentile_image(gray)
        ids = ids_from_mask(np.asarray(Image.open(mask_path).convert("RGB")))
        group = groups[specimen_id(image_path)]
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for class_id in range(4):
            locations = np.flatnonzero(ids == class_id)
            if len(locations) == 0:
                continue
            take = rng.choice(locations, min(len(locations), 10000), replace=False)
            group["z"][class_id].append(z.reshape(-1)[take])
            group["q"][class_id].append(q.reshape(-1)[take])

    result = {}
    for group_name, values in groups.items():
        result[group_name] = {}
        for key in ("z", "q"):
            stats = []
            for class_values in values[key]:
                if not class_values:
                    stats.append(None)
                    continue
                merged = np.concatenate(class_values)
                stats.append((float(merged.mean()), max(float(merged.std()), 0.15 if key == "z" else 0.02)))
            result[group_name][key] = stats
    return result


def blend_stats(global_stats, group_stats, rho, floor):
    if group_stats is None:
        return list(global_stats)
    output = []
    for global_value, group_value in zip(global_stats, group_stats):
        if group_value is None:
            output.append(global_value)
            continue
        mean = (1.0 - rho) * global_value[0] + rho * group_value[0]
        std = max((1.0 - rho) * global_value[1] + rho * group_value[1], floor)
        output.append((float(mean), float(std)))
    return output


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data = Path(args.data_dir)
    global_z = fit_global_stats(data)
    global_q = fit_percentile_stats(data)
    group_stats = fit_group_stats(data)
    train_values = []
    for image_path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_values.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([value[0] for value in train_values]))
    train_std = float(np.mean([value[1] for value in train_values]))
    pair = load_aspp(args.pair_checkpoint, device)

    source_paths = {
        "restart51": args.source_restart51,
        "model109": args.source_model109,
        "seed456": args.source_seed456,
        "pairwise": args.source_pairwise,
        "ema": args.source_ema,
    }
    source_args = argparse.Namespace(**{f"source_{key}": value for key, value in source_paths.items()})
    entries = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, source_args)
        source_name = next(name for name, path in source_paths.items() if path == source)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        style = transform_image(raw, "standardize", train_mean, train_std)
        high768_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high768_style = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")
        high1024_raw = predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian")
        high1024_style = predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian")
        alpha = args.alpha_1024
        high = tuple((1.0 - alpha) * old + alpha * new for old, new in zip(
            (high768_raw, high768_style), (high1024_raw, high1024_style)))
        entries.append({
            "name": image_path.name,
            "specimen": specimen_id(image_path),
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32),
            "high": tuple(value.astype(np.float32) for value in high),
            "z": (gray - mean) / max(std, 1e-5),
            "q": percentile_image(gray),
            "global_z": global_z,
            "global_q": global_q,
            "group": group_stats.get(specimen_id(image_path)),
            "params": route_params(mean, std),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
        del adapted, high768_raw, high768_style, high1024_raw, high1024_style
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name} source={source_name} group={entries[-1]['group'] is not None}", flush=True)

    records = []
    for rho in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0):
        for beta_scale in (0.0, 0.50, 0.75, 1.0, 1.25, 1.50):
            predictions, targets = [], []
            for entry in entries:
                group = entry["group"]
                group_z = group["z"] if group is not None else None
                group_q = group["q"] if group is not None else None
                item = dict(entry)
                item["prior_z_stats"] = blend_stats(entry["global_z"], group_z, rho, 0.15)
                item["prior_q_stats"] = blend_stats(entry["global_q"], group_q, rho, 0.02)
                item["params"] = dict(entry["params"])
                item["params"]["beta"] *= beta_scale
                logits = make_logits(item, entry["high"])
                predictions.append(predict(logits))
                targets.append(entry["target"])
            value = metrics(predictions, targets)
            row = {"alpha_1024": args.alpha_1024, "rho": rho,
                   "beta_scale": beta_scale, **value}
            records.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    best = max(records, key=lambda row: row["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--source-restart51", required=True)
    parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-seed456", required=True)
    parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-ema", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--alpha-1024", type=float, default=0.20)
    main(parser.parse_args())
