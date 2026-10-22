#!/usr/bin/env python3
"""Evaluate Model257 with single-pass whole-image D4 inference on valid only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import inverse_view, transform_view, views
from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import representation, score
from model239_native_stem_finetune import Stride1NativeStemASPP


@torch.inference_mode()
def predict_whole_image(model, image_path: Path, device: torch.device,
                        use_amp: bool) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    features = representation(image)
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    if padded_height != height or padded_width != width:
        features = np.pad(features, ((0, 0), (0, padded_height - height),
                                     (0, padded_width - width)), mode="reflect")
    inputs = torch.from_numpy(features).unsqueeze(0).to(device)
    probabilities = None
    for rotation, flip in views("d4"):
        view = transform_view(inputs, rotation, flip)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            current = torch.softmax(model(view), dim=1)
        current = inverse_view(current, rotation, flip)
        probabilities = current if probabilities is None else probabilities + current
    result = (probabilities / 8.0)[0, :, :height, :width].float().cpu().numpy()
    result /= np.maximum(result.sum(axis=0, keepdims=True), 1e-7)
    return result


def main(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    samples = sorted(samples, key=lambda pair: pair[0].name)
    if args.image_name:
        samples = [pair for pair in samples if pair[0].name == args.image_name]
        if not samples:
            raise FileNotFoundError(f"valid image not found: {args.image_name}")
    if args.limit:
        samples = samples[:args.limit]

    model = Stride1NativeStemASPP().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded checkpoint strictly: {args.checkpoint}", flush=True)
    predictions, targets, records = [], [], []
    probability_payload = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for image_path, mask_path in samples:
        image_start = time.time()
        probability = predict_whole_image(model, image_path, device, use_amp)
        prediction = probability.argmax(axis=0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(prediction)
        targets.append(target)
        probability_payload[image_path.stem] = probability.astype(np.float16)
        records.append({"image": image_path.name, "height": int(target.shape[0]),
                        "width": int(target.shape[1]),
                        "seconds": time.time() - image_start})
        print(json.dumps({"processed": len(records), "total": len(samples),
                          "image": image_path.name, "seconds": records[-1]["seconds"]},
                         ensure_ascii=False), flush=True)

    metrics = score(predictions, targets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "valid_probs.npz", **probability_payload)
    peak_memory = (int(torch.cuda.max_memory_allocated(device))
                   if device.type == "cuda" else None)
    summary = {
        "model": "Model257 ResNet34 U-Net++/ASPP + Model239 native stem",
        "inference": "whole image, scale=1, full-image raw/z/CDF representation, D4 probability mean",
        "split": "valid",
        "count": len(samples),
        "metrics": metrics,
        "images": records,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_allocated_bytes": peak_memory,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json"),
                      "image_mean_miou": metrics["image_mean_miou"],
                      "global_miou": metrics["global_miou"],
                      "peak_cuda_allocated_bytes": peak_memory,
                      "elapsed_seconds": summary["elapsed_seconds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-name", default="",
                        help="evaluate one named valid image for a memory smoke test")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N valid images; 0 means all")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
