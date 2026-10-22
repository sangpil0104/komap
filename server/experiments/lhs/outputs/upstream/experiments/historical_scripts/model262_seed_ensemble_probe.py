#!/usr/bin/env python3
"""Average full-image-stat seed probabilities and score on validation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model181_raw_zcdf_aspp import score


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    first = np.load(args.seed257)
    second = np.load(args.seed261)
    if tuple(first.files) != tuple(second.files) or len(first.files) != len(entries):
        raise ValueError("seed probability archives must have matching valid-image order")

    probabilities = {}
    predictions, targets = [], []
    for index, key in enumerate(first.files):
        mean = (first[key].astype(np.float32) + second[key].astype(np.float32)) * 0.5
        mean /= np.maximum(mean.sum(axis=0, keepdims=True), 1e-7)
        probabilities[key] = mean.astype(np.float16)
        predictions.append(mean.argmax(axis=0).astype(np.uint8))
        targets.append(entries[index]["target"])
    metrics = score(predictions, targets)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "valid_probs.npz", **probabilities)
    (output / "summary.json").write_text(json.dumps({
        "method": "equal probability average of full-image-stat seeds 257 and 261",
        "sources": [args.seed257, args.seed261],
        "evaluation": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--seed257", required=True)
    parser.add_argument("--seed261", required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
