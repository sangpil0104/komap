#!/usr/bin/env python3
"""Save held-out-train probability maps only; never open masks or calculate scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
import sys
sys.path.insert(0, str(SOURCE_ROOT))

from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing  # noqa: E402
from komap_seg.dataset import (  # noqa: E402
    KomapFullDataset,
    effective_input_scale,
    scaled_overlap,
)
from komap_seg.inference import predict_probs_tta  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--overlap", type=int, default=170)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty Model410 prediction output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_dir = args.data_root / "valid/images"
    # Intentionally construct the image-only dataset: held-out masks are never
    # opened here, and no metric calculation or per-image score is performed.
    dataset = KomapFullDataset(images_dir)
    if not len(dataset):
        raise ValueError(f"no held-out images found in {images_dir}")
    device = torch.device(args.device)
    model, saved_preprocessing = load_model_and_preprocessing(args.checkpoint, device)
    preprocessing = resolve_preprocessing(
        saved_preprocessing, args.patch_size, args.input_scale, None
    )

    probabilities: dict[str, np.ndarray] = {}
    for sample in dataset:
        scale = effective_input_scale(
            preprocessing["input_scale"], preprocessing["target_width"], sample["width"]
        )
        overlap = scaled_overlap(
            preprocessing["patch_size"], preprocessing["input_scale"], args.overlap, scale
        )
        probability = predict_probs_tta(
            model,
            sample["image"],
            patch_size=preprocessing["patch_size"],
            overlap=overlap,
            batch_size=args.batch_size,
            tta=True,
            input_scale=scale,
            device=device,
            tta_mode="flip4",
            downsample_antialias=False,
        ).numpy().astype(np.float32, copy=False)
        if probability.ndim != 3 or probability.shape[0] != 4 or not np.isfinite(probability).all():
            raise ValueError(f"invalid model probability map for {sample['stem']}")
        probabilities[sample["stem"]] = probability.astype(np.float16)

    archive_path = output / "valid_probabilities.npz"
    np.savez_compressed(archive_path, **probabilities)
    manifest = {
        "experiment": "Model410 held-out train prediction-only inference",
        "images": len(probabilities),
        "image_ids": sorted(probabilities),
        "checkpoint_sha256": sha256(args.checkpoint),
        "inference": {
            "patch_size": preprocessing["patch_size"],
            "input_scale": preprocessing["input_scale"],
            "target_width": preprocessing["target_width"],
            "overlap_base": args.overlap,
            "tta": "flip4",
            "batch_size": args.batch_size,
        },
        "masks_opened": False,
        "metrics_computed": False,
        "official_valid_or_test_used": False,
    }
    with (output / "inference_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"saved {len(probabilities)} held-out train probability maps; no scores computed", flush=True)


if __name__ == "__main__":
    main()
