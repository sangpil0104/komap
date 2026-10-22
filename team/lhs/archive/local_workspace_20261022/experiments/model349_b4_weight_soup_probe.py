#!/usr/bin/env python3
"""Build a fold-matched B4 weight interpolation for a compression probe.

This only creates a candidate checkpoint; it does not claim equivalence to
averaging two models' prediction probabilities. Run held-out fold inference
with the existing B4 predict.py before using the checkpoint anywhere else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--alpha329", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-config", choices=("326", "329"), default="329")
    args = parser.parse_args()
    if not 0.0 < args.alpha329 < 1.0:
        raise ValueError("alpha329 must be strictly between zero and one")
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root / "outputs"
    path326 = root / f"model326_hybrid_fullstats_group_oof/fold{args.fold}"
    path329 = root / f"model329_fullstats_width4000_uniform_group_oof/fold{args.fold}"
    state326 = torch.load(path326 / "last.pt", map_location="cpu", weights_only=True)
    state329 = torch.load(path329 / "last.pt", map_location="cpu", weights_only=True)
    if state326.keys() != state329.keys():
        raise ValueError("checkpoint state_dict keys differ")
    merged = {}
    for key, value326 in state326.items():
        value329 = state329[key]
        if value326.shape != value329.shape or value326.dtype != value329.dtype:
            raise ValueError(f"incompatible checkpoint tensor: {key}")
        merged[key] = (
            torch.lerp(value326, value329, args.alpha329)
            if value326.is_floating_point() else value329.clone()
        )
    source = path326 if args.input_config == "326" else path329
    config = json.loads((source / "args.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    torch.save(merged, args.output / "last.pt")
    (args.output / "args.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "fold": args.fold,
        "alpha329": args.alpha329,
        "checkpoint326": str(path326 / "last.pt"),
        "checkpoint329": str(path329 / "last.pt"),
        "input_config": args.input_config,
        "purpose": "held-out OOF compression probe only",
    }
    (args.output / "soup_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
