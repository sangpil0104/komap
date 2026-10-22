"""Predict phase masks for every `*_image.png` in a folder and save them as
color-coded `{stem}_mask.png` files in the competition palette.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
from komap_seg.inference import predict_probs_tta
from komap_seg.palette import class_to_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", nargs="+", required=True, help="One or more .pt files; several are ensembled.")
    p.add_argument("--images-dir", required=True, help="e.g. <data-root>/test/images")
    p.add_argument("--output-dir", default="runs/best/predictions")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=170)
    p.add_argument("--input-scale", type=float, default=None, help="Optional assertion against checkpoint metadata.")
    p.add_argument("--target-width", type=int, default=None,
                   help="Optional assertion against checkpoint metadata; normally inferred per checkpoint.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--tta-mode", choices=("flip4", "d4"), default="flip4")
    p.add_argument("--downsample-antialias", action="store_true",
                   help="low-pass filter probability maps when reducing scaled tiles")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for checkpoint_path in args.checkpoint:
        model, saved_preprocessing = load_model_and_preprocessing(Path(checkpoint_path), device)
        preprocessing = resolve_preprocessing(
            saved_preprocessing, args.patch_size, args.input_scale, args.target_width
        )
        models.append((model, preprocessing))
    ds = KomapFullDataset(Path(args.images_dir), masks_dir=None)

    for sample in ds:
        probs = None
        for model, preprocessing in models:
            sample_scale = effective_input_scale(
                preprocessing["input_scale"], preprocessing["target_width"], sample["width"]
            )
            sample_overlap = scaled_overlap(
                preprocessing["patch_size"], preprocessing["input_scale"],
                args.overlap, sample_scale
            )
            model_probs = predict_probs_tta(
                model,
                sample["image"],
                patch_size=preprocessing["patch_size"],
                overlap=sample_overlap,
                batch_size=args.batch_size,
                tta=not args.no_tta,
                input_scale=sample_scale,
                device=device,
                tta_mode=args.tta_mode,
                downsample_antialias=args.downsample_antialias,
            )
            probs = model_probs if probs is None else probs + model_probs
        rgb = class_to_rgb(probs.argmax(0).numpy().astype(np.uint8))
        out_path = output_dir / f"{sample['stem']}_mask.png"
        Image.fromarray(rgb).save(out_path)
        print(f"{sample['stem']}: saved {out_path}", flush=True)

    print(f"done. {len(ds)} masks in {output_dir}")


if __name__ == "__main__":
    main()
