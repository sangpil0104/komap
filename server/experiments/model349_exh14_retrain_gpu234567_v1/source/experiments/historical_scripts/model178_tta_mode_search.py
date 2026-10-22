#!/usr/bin/env python3
"""Compare raw/style/joint source-free TENT in the model-173 route.

The route, source selection, adaptive intensity prior, cleanup, and 768/1024
blend are kept fixed.  Only the unlabeled adaptation inputs are changed:
raw-domain tiles, standardized-style tiles, or both domains jointly.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from hard_source_search import route_params
from highres1024_search import choose_source, make_logits, predict
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from domain_tent_search import adapt_domain


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data = Path(args.data_dir)

    train_stats = []
    for image_path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([value[0] for value in train_stats]))
    train_std = float(np.mean([value[1] for value in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)

    source_paths = {
        "restart51": args.source_restart51,
        "model109": args.source_model109,
        "seed456": args.source_seed456,
        "pairwise": args.source_pairwise,
        "ema": args.source_ema,
    }
    samples = collect_split(data, "valid")
    entries = []
    for image_path, mask_path in _local_progress(samples, 'model178_tta_mode_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, argparse.Namespace(
            source_pairwise=source_paths["pairwise"],
            source_ema=source_paths["ema"],
            source_model109=source_paths["model109"],
            source_seed456=source_paths["seed456"],
            source_restart51=source_paths["restart51"],
        ))
        # choose_source returns a path; preserve a readable source label too.
        source_name = next(name for name, path in source_paths.items() if path == source)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        entries.append({
            "name": image_path.name,
            "source": source,
            "source_name": source_name,
            "raw": raw,
            "style": style,
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32),
            "z": (gray - mean) / max(std, 1e-5),
            "q": percentile_image(gray),
            "prior_z_stats": z_stats,
            "prior_q_stats": q_stats,
            "params": route_params(mean, std),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
        print(f"cached {image_path.name} source={source_name}", flush=True)

    records = []
    for mode in ("raw", "style", "joint"):
        for entry in entries:
            adapted = adapt_domain(entry["source"], entry["raw"], entry["style"], mode,
                                   device, steps=args.steps, lr=args.lr)
            raw768 = predict_array(adapted, entry["raw"], 768, 576, device, amp, 1, "d4", "gaussian")
            style768 = predict_array(adapted, entry["style"], 768, 576, device, amp, 1, "d4", "gaussian")
            raw1024 = predict_array(adapted, entry["raw"], 1024, 768, device, amp, 1, "d4", "gaussian")
            style1024 = predict_array(adapted, entry["style"], 1024, 768, device, amp, 1, "d4", "gaussian")
            entry.setdefault("branches_by_mode", {})[mode] = {
                "high768": (raw768.astype(np.float32), style768.astype(np.float32)),
                "high1024": (raw1024.astype(np.float32), style1024.astype(np.float32)),
            }
            del adapted, raw768, style768, raw1024, style1024
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"adapted mode={mode} image={entry['name']}", flush=True)

        for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0):
            predictions, targets = [], []
            for entry in entries:
                high768 = entry["branches_by_mode"][mode]["high768"]
                high1024 = entry["branches_by_mode"][mode]["high1024"]
                high = tuple((1.0 - alpha) * high768[index] + alpha * high1024[index]
                             for index in (0, 1))
                predictions.append(predict(make_logits(entry, high)))
                targets.append(entry["target"])
            row = {"mode": mode, "steps": args.steps, "lr": args.lr,
                   "alpha_1024": alpha, **metrics(predictions, targets)}
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
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    main(parser.parse_args())
