#!/usr/bin/env python3
"""Check whether Model374's 2x label/probability resize round-trip loses pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def run() -> dict[str, Any]:
    cases = []
    for seed, shape in ((1, (256, 256)), (2, (256, 256)), (3, (128, 256)), (4, (255, 255))):
        generator = torch.Generator().manual_seed(seed)
        target = torch.randint(0, 4, (1, *shape), generator=generator)
        one_hot = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2).to(torch.float32)
        expanded_target = F.interpolate(
            one_hot, size=(shape[0] * 2, shape[1] * 2), mode="nearest"
        )
        returned_probabilities = F.interpolate(
            expanded_target, size=shape, mode="bilinear", align_corners=False
        )
        prediction = returned_probabilities.argmax(dim=1)
        errors = int((prediction != target).sum().item())
        non_one_hot = int((returned_probabilities.max(dim=1).values < 0.999).sum().item())
        cases.append({
            "seed": seed,
            "native_shape": list(shape),
            "native_pixels": shape[0] * shape[1],
            "argmax_mismatches": errors,
            "non_one_hot_native_pixels": non_one_hot,
        })
        if errors != 0 or non_one_hot != 0:
            raise AssertionError(f"2x one-hot round-trip was not exact: {cases[-1]}")

    return {
        "model": 386,
        "scope": "synthetic transform-only check; does not test encoder/decoder feature stride or learned logits",
        "chain": "native one-hot labels -> nearest 2x -> ideal one-hot probabilities -> bilinear 0.5x align_corners=False",
        "cases": cases,
        "all_exact": True,
        "interpretation_guard": "The pixel-resize round-trip itself adds no label loss under ideal one-hot predictions; this does not rule out learned feature-resolution or context limitations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
