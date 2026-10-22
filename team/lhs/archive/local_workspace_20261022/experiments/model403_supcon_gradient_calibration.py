#!/usr/bin/env python3
"""Measure Model403 auxiliary-vs-supervised decoder-gradient scale on outer-train crops.

This is a pre-training diagnostic only: it reads one fixed OOF fold's train
images/masks, computes gradients, and never steps an optimizer.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
sys.path.insert(0, str(SOURCE_ROOT))

from komap_seg.dataset import KomapPatchDataset, modal_image_width  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from komap_seg.model import build_model  # noqa: E402
from model403_local_supcon_train import (  # noqa: E402
    build_projection_head,
    supervised_local_contrastive,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(data_root: Path, device: torch.device, seed: int, weight: float) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_root = data_root.resolve()
    images_dir = data_root / "train/images"
    masks_dir = data_root / "train/masks"
    width = modal_image_width(images_dir)
    dataset = KomapPatchDataset(
        images_dir,
        masks_dir,
        patch_size=512,
        input_scale=2.0,
        target_width=width,
        samples_per_image=12,
        seed=seed,
        rng_mode="legacy",
    )
    set_random_seed = getattr(dataset.transform, "set_random_seed", None)
    if not callable(set_random_seed):
        raise RuntimeError("reproducible probe requires Albumentations Compose.set_random_seed")
    set_random_seed(seed)

    model_config = {
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet",
        "boundary_head": True,
    }
    model = build_model(**model_config).to(device).train()
    projector = build_projection_head(model.segmentation_head[0].in_channels, device).train()
    segmentation_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    rows: list[dict[str, float | int | list[str]]] = []

    with torch.enable_grad():
        for batch_no in range(4):
            image_indices = [batch_no * 4 + offset for offset in range(4)]
            dataset_indices = [index * dataset.samples_per_image for index in image_indices]
            if max(dataset_indices) >= len(dataset):
                raise ValueError("fixed 16-image probe exceeds this fold's training inventory")
            pairs = [dataset[index] for index in dataset_indices]
            images = torch.stack([pair[0] for pair in pairs]).to(device)
            masks = torch.stack([pair[1] for pair in pairs]).to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                decoded = model.decoder(model.encoder(images))
                loss_seg = segmentation_loss(model.segmentation_head(decoded), masks)
                loss_boundary = boundary_loss(model.boundary_head(decoded), masks)
                loss_supcon = supervised_local_contrastive(
                    projector(decoded), masks, samples_per_class=16, temperature=0.1
                )
            supervised_grad = torch.autograd.grad(
                loss_seg + loss_boundary, decoded, retain_graph=True
            )[0].float()
            auxiliary_grad = torch.autograd.grad(weight * loss_supcon, decoded)[0].float()
            supervised_norm = float(torch.linalg.vector_norm(supervised_grad).item())
            auxiliary_norm = float(torch.linalg.vector_norm(auxiliary_grad).item())
            ratio = auxiliary_norm / max(supervised_norm, 1e-12)
            rows.append({
                "batch": batch_no + 1,
                "image_ids": [dataset.stems[index] for index in image_indices],
                "seg_loss": float(loss_seg.item()),
                "boundary_loss": float(loss_boundary.item()),
                "supcon_loss_unweighted": float(loss_supcon.item()),
                "supervised_decoder_grad_norm": supervised_norm,
                "weighted_supcon_decoder_grad_norm": auxiliary_norm,
                "gradient_norm_ratio": ratio,
            })
            if not all(np.isfinite(value) for value in (supervised_norm, auxiliary_norm, ratio)):
                raise FloatingPointError(f"non-finite gradient measurement in batch {batch_no+1}")
            del images, masks, decoded, supervised_grad, auxiliary_grad, pairs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ratios = [float(row["gradient_norm_ratio"]) for row in rows]
    median_ratio = statistics.median(ratios)
    return {
        "experiment": "Model403 supervised contrastive loss-scale preflight",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "fixed outer-train fold only; no held-out fold, official valid, or test read",
        "data_root": str(data_root),
        "fold_train_images": len(dataset.images),
        "probe_images": [stem for row in rows for stem in row["image_ids"]],
        "seed": seed,
        "device": str(device),
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "model": model_config,
        "patch_size": 512,
        "input_scale": 2.0,
        "target_width": width,
        "batch_size": 4,
        "contrastive_weight_tested": weight,
        "contrastive_temperature": 0.1,
        "positions_per_class_per_image": 16,
        "decoder_gradient_norm_ratio": {
            "values_by_batch": ratios,
            "min": min(ratios),
            "median": median_ratio,
            "max": max(ratios),
            "weight_for_unit_median_ratio": weight / median_ratio,
        },
        "batches": rows,
        "optimizer_steps": 0,
        "official_valid_read": False,
        "test_read": False,
        "trainer_sha256": sha256_file(ROOT / "experiments/model403_local_supcon_train.py"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--weight", type=float, default=0.0062)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = measure(args.data_root, torch.device(args.device), args.seed, args.weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result["decoder_gradient_norm_ratio"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
