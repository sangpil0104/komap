#!/usr/bin/env python3
"""Transfer a self-supervised encoder into a supervised segmentation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PREFIXES = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    ssl = torch.load(args.ssl_checkpoint, map_location="cpu", weights_only=False)
    model_state = base["model"]
    encoder_state = ssl["encoder"]
    expected = {key for key in model_state if key.startswith(PREFIXES)}
    if expected != set(encoder_state):
        missing, extra = expected - set(encoder_state), set(encoder_state) - expected
        raise ValueError(f"encoder key mismatch: missing={sorted(missing)[:8]}, "
                         f"extra={sorted(extra)[:8]}")
    mismatched = [key for key in expected
                  if model_state[key].shape != encoder_state[key].shape]
    if mismatched:
        raise ValueError(f"encoder shape mismatch: {mismatched[:8]}")

    transferred = dict(model_state)
    transferred.update(encoder_state)
    base["model"] = transferred
    base["initialization"] = {
        "base_checkpoint": str(args.base_checkpoint),
        "ssl_checkpoint": str(args.ssl_checkpoint),
        "ssl_epoch": ssl.get("epoch"),
        "ssl_best_loss": ssl.get("best_loss"),
        "transferred_keys": len(expected),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output)
    print(base["initialization"], flush=True)


if __name__ == "__main__":
    main()
