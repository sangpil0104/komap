"""Predict phase masks for every `*_image.png` in a folder and save them as
color-coded `{stem}_mask.png` files in the competition palette.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from komap_seg.checkpoint import load_model
from komap_seg.dataset import KomapFullDataset
from komap_seg.inference import predict_probs_tta
from komap_seg.palette import class_to_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", nargs="+", required=True, help="One or more .pt files; several are ensembled.")
    p.add_argument("--images-dir", required=True, help="e.g. <data-root>/test/images")
    p.add_argument("--output-dir", default="runs/best/predictions")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=170)
    p.add_argument("--input-scale", type=float, default=2.0, help="Must match training.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--tta-mode", choices=("flip4", "d4"), default="flip4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [load_model(Path(c), device) for c in args.checkpoint]
    ds = KomapFullDataset(Path(args.images_dir), masks_dir=None)

    for sample in ds:
        probs = sum(
            predict_probs_tta(
                m,
                sample["image"],
                patch_size=args.patch_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                tta=not args.no_tta,
                input_scale=args.input_scale,
                device=device,
                tta_mode=args.tta_mode,
            )
            for m in models
        )
        rgb = class_to_rgb(probs.argmax(0).numpy().astype(np.uint8))
        out_path = output_dir / f"{sample['stem']}_mask.png"
        Image.fromarray(rgb).save(out_path)
        print(f"{sample['stem']}: saved {out_path}", flush=True)

    print(f"done. {len(ds)} masks in {output_dir}")


if __name__ == "__main__":
    main()
