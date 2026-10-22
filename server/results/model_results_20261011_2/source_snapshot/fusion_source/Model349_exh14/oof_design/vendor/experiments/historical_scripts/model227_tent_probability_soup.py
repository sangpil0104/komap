#!/usr/bin/env python3
"""Average per-image probabilities from two native TENT adaptation lengths."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, help="shorter-step TENT NPZ")
    parser.add_argument("--second", required=True, help="longer-step TENT NPZ")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight assigned to the second probability set")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    first, second = np.load(args.first), np.load(args.second)
    if first.files != second.files:
        raise ValueError("NPZ keys/order do not match")
    output: dict[str, np.ndarray] = {}
    max_sum_error = 0.0
    for key in first.files:
        p = first[key].astype(np.float32)
        q = second[key].astype(np.float32)
        if p.shape != q.shape or p.ndim != 3:
            raise ValueError(f"invalid probability shape for {key}: {p.shape}, {q.shape}")
        value = (1.0 - args.alpha) * p + args.alpha * q
        value /= np.maximum(value.sum(axis=0, keepdims=True), 1e-7)
        max_sum_error = max(max_sum_error,
                            float(np.abs(value.sum(axis=0) - 1.0).max()))
        output[key] = value.astype(np.float16)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    print({"images": len(output), "alpha": args.alpha,
           "max_probability_sum_error": max_sum_error,
           "output": str(path)})


if __name__ == "__main__":
    main()
