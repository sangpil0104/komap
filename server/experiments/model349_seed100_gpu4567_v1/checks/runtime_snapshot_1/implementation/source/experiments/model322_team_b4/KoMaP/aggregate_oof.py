#!/usr/bin/env python3
"""Combine held-out fold probability archives and verify exactly-once OOF coverage."""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=70)
    args = parser.parse_args()

    combined = {}
    for fold in range(1, args.folds + 1):
        path = args.runs / f"fold{fold}_oof" / "probabilities.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            for name in archive.files:
                if name in combined:
                    raise ValueError(f"duplicate OOF prediction: {name}")
                combined[name] = archive[name]
    if len(combined) != args.expected_images:
        raise ValueError(f"expected {args.expected_images} OOF images, got {len(combined)}")
    output = args.runs / "train_oof_probs.npz"
    np.savez_compressed(output, **combined)
    print(f"saved {len(combined)} exactly-once OOF predictions: {output}", flush=True)


if __name__ == "__main__":
    main()
