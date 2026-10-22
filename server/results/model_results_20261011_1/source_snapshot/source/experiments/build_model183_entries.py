#!/usr/bin/env python3
"""Regenerate the Model183 routing entries from newly trained checkpoints."""

import argparse
from pathlib import Path

import torch

from model183_metric_calibration import cache_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir", "cache-dir", "representation-checkpoint", "pair-checkpoint",
        "source-restart51", "source-model109", "source-seed456",
        "source-pairwise", "source-ema", "output-json",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--alpha-1024", type=float, default=0.20)
    args = parser.parse_args()
    for name in (
        "representation_checkpoint", "pair_checkpoint", "source_restart51",
        "source_model109", "source_seed456", "source_pairwise", "source_ema",
    ):
        path = Path(getattr(args, name))
        if not path.is_file():
            parser.error(f"missing trained checkpoint: {path}")
    entries = cache_entries(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Model183 route entries: {len(entries)} validation images", flush=True)


if __name__ == "__main__":
    main()
