#!/usr/bin/env python3
"""Export validation probabilities from a saved TeamRepresentationASPP checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from improve_five_causes import collect_split
from model181_raw_zcdf_aspp import TeamRepresentationASPP, evaluate, load_checkpoint


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint)
    samples = collect_split(args.data_dir, "valid")
    eval_args = argparse.Namespace(
        patch_size=args.patch_size,
        val_stride=args.val_stride,
        tile_batch_size=args.tile_batch_size,
    )
    result = evaluate(model, samples, eval_args, device,
                      device.type == "cuda", output_probs=True)
    probabilities = result.pop("probabilities")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {path.stem: value.astype(np.float16)
               for (path, _), value in zip(samples, probabilities)}
    np.savez_compressed(output / "valid_probs.npz", **payload)
    (output / "export_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    main(parser.parse_args())
