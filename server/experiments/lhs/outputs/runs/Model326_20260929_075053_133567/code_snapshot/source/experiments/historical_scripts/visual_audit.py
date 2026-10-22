#!/usr/bin/env python3
"""Create visual QA sheets for the hardest validation images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from calibrate_al import predict_probabilities
from improve_five_causes import CLASS_COLORS, collect_split, decode_mask, encode_mask
from hard_negative_finetune import load_model


def color_mask(mask: np.ndarray) -> Image.Image:
    return encode_mask(mask).convert("RGB")


def make_panel(image: Image.Image, target: Image.Image, prediction: Image.Image, title: str, width: int) -> Image.Image:
    height = int(round(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    target = target.resize((width, height), Image.Resampling.NEAREST)
    prediction = prediction.resize((width, height), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (width * 3, height + 34), "white")
    panel.paste(image, (0, 34))
    panel.paste(target, (width, 34))
    panel.paste(prediction, (width * 2, 34))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill="black")
    draw.text((8, 20), "input", fill="black")
    draw.text((width + 8, 20), "ground truth", fill="black")
    draw.text((width * 2 + 8, 20), "prediction", fill="black")
    return panel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    samples = collect_split(args.data_dir, "valid")
    wanted = set(args.names.split(",")) if args.names else None
    panels = []
    for image_path, mask_path in samples:
        if wanted and image_path.name not in wanted:
            continue
        image = Image.open(image_path).convert("RGB")
        target_np = decode_mask(Image.open(mask_path).convert("RGB"))
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.stride, device,
                                               use_amp, args.tile_batch_size, True)
        adjusted = np.log(np.maximum(probabilities, 1e-7))
        adjusted[3] += args.al_bias
        prediction = adjusted.argmax(axis=0).astype(np.uint8)
        panels.append(make_panel(image, color_mask(target_np), color_mask(prediction), image_path.name, args.width))
    if not panels:
        raise SystemExit("No matching validation images")
    sheet = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"wrote {len(panels)} panels to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--names", default="")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--al-bias", type=float, default=-0.60)
    parser.add_argument("--width", type=int, default=440)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
