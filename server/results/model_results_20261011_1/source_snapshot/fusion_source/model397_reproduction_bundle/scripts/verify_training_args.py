#!/usr/bin/env python3
"""Refuse to reuse a checkpoint whose saved recipe differs from Model397."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED = {
    "model329": {
        "arch": "unet", "encoder": "efficientnet-b4", "norm": "image_global",
        "input": "zcdf3", "crop": 1536, "scale": 1.0, "target_width": 4000,
        "sample": "uniform", "bs": 2, "iters": 16000, "lr": 3e-4,
        "wd": 1e-2, "dice_w": 1.0, "cldice_w": 0.0, "seed": 0,
        "workers": 4, "no_valid_eval": True,
    },
    "model392": {
        "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
        "normalize_width": True, "samples_per_image": 12, "batch_size": 4,
        "accumulation_steps": 2, "num_workers": 4, "epochs": 70,
        "lr": 3e-4, "encoder_lr_mult": 1.0, "weight_decay": 1e-4,
        "boundary_head_weight": 1.0, "rng_mode": "legacy", "val_every": 0,
        "no_amp": False,
    },
}


def same(actual, expected) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--args-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    saved = json.loads(args.args_file.read_text(encoding="utf-8"))
    expected = dict(EXPECTED[args.family])
    if args.family == "model329":
        expected["train_splits"] = f"train_fold{args.fold}"
        expected["data"] = str((args.args_file.parent.parent.parent / "splits/model329_cv").resolve())
        expected["out"] = str(args.args_file.parent.resolve())
        if Path(saved.get("data", "")).resolve() != Path(expected["data"]):
            raise SystemExit("saved B4 data root is not the prepared model329_cv split")
    else:
        expected["seed"] = args.seed
        expected["data_root"] = str((args.args_file.parent.parent.parent.parent
                                      / f"splits/model392_cv/seed{args.seed}_fold{args.fold}").resolve())
        expected["output_dir"] = str(args.args_file.parent.resolve())
        if Path(saved.get("data_root", "")).resolve() != Path(expected["data_root"]):
            raise SystemExit("saved HRNet data root does not match this seed/fold")
    mismatches = {
        key: {"expected": value, "actual": saved.get(key)}
        for key, value in expected.items()
        if not same(saved.get(key), value)
    }
    if mismatches:
        raise SystemExit(json.dumps({"status": "RECIPE_MISMATCH", "mismatches": mismatches}, indent=2))
    print(f"verified {args.family} seed={args.seed} fold={args.fold}: {args.args_file}")


if __name__ == "__main__":
    main()
