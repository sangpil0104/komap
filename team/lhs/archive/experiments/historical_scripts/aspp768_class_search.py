#!/usr/bin/env python3
"""Class-specific 768-member weight search using cached validation probabilities."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from aspp768_addition import BIASES, MATRIX, cleanup_primary
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


def score(cached, weights, threshold=128):
    predictions, targets = [], []
    weights = np.asarray(weights, dtype=np.float32)
    for values, prior, target in cached:
        mixed = np.asarray([
            sum(MATRIX[c, m] * (1.0 - weights[c]) * values[m][c] for m in range(3))
            + weights[c] * values[3][c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior + BIASES[:, None, None]
        predictions.append(cleanup_primary(logits.argmax(axis=0).astype(np.uint8), logits, threshold))
        targets.append(target)
    return metrics(predictions, targets)


def main(args):
    cache_dir = Path(args.cache_dir)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        loaded = np.load(cache_dir / f"{image_path.stem}.npz")
        values = [loaded[f"p{i}"].astype(np.float32) for i in range(4)]
        prior = loaded["prior"].astype(np.float32)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append((values, prior, target))

    candidate_values = (0.3, 0.45, 0.6, 0.75, 0.9)
    current = np.full(4, 0.6, dtype=np.float32)
    history = []
    for round_index in range(3):
        improved = False
        for class_id in range(4):
            candidates = []
            for value in candidate_values:
                proposal = current.copy(); proposal[class_id] = value
                result = score(cached, proposal, args.threshold)
                candidates.append((result["image_mean_miou"], result["global_miou"], value, result))
            best = max(candidates, key=lambda item: item[0])
            if best[2] != current[class_id]:
                improved = True
            current[class_id] = best[2]
            history.append({"round": round_index, "class_id": class_id,
                            "weights": current.tolist(), "selected": best[2],
                            "result": best[3]})
            print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if not improved:
            break

    final = score(cached, current, args.threshold)
    result = {"weights": current.tolist(), "threshold": args.threshold, "final": final, "history": history}
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"weights": current.tolist(), "threshold": args.threshold,
                      "global_miou": final["global_miou"],
                      "image_mean_miou": final["image_mean_miou"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--threshold", type=int, default=128); main(parser.parse_args())
