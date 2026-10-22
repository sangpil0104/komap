#!/usr/bin/env python3
"""Outer-train-only, no-optimizer-step scale calibration for Model404."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
sys.path.insert(0, str(SOURCE_ROOT))

from komap_seg.checkpoint import load_model_and_preprocessing  # noqa: E402
from komap_seg.dataset import KomapPatchDataset, modal_image_width, seed_worker  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from model404_component_recall_loss import build_component_targets, component_recall_loss  # noqa: E402


def calibrate_fold(data_root: Path, checkpoint: Path, device: torch.device,
                   probe_batches: int, probe_seed: int) -> dict:
    random.seed(probe_seed)
    np.random.seed(probe_seed)
    torch.manual_seed(probe_seed)
    model, preprocessing = load_model_and_preprocessing(checkpoint, device)
    target_width = modal_image_width(data_root / "train" / "images")
    if target_width != preprocessing["target_width"]:
        raise ValueError(f"train mode width differs from checkpoint: {target_width}")
    dataset = KomapPatchDataset(
        data_root / "train" / "images",
        data_root / "train" / "masks",
        patch_size=512,
        input_scale=2.0,
        target_width=target_width,
        samples_per_image=12,
        seed=1,
        rng_mode="per_item",
    )
    dataset.set_epoch(1)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    seg_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    ratios: list[float] = []
    component_counts: list[int] = []
    model.train()
    for batch_index, (images_cpu, masks_cpu) in enumerate(loader):
        if batch_index >= probe_batches:
            break
        components = build_component_targets(masks_cpu)
        if components.count == 0:
            component_counts.append(0)
            continue
        masks = masks_cpu.to(device, non_blocking=True)
        images = images_cpu.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoded = model.encoder(images)
            decoded = model.decoder(encoded)
            logits = model.segmentation_head(decoded)
            boundary_logits = model.boundary_head(decoded)
            main = seg_loss(logits, masks) + boundary_loss(boundary_logits, masks)
            auxiliary = component_recall_loss(logits, components)
        main_grad = torch.autograd.grad(main, decoded, retain_graph=True)[0]
        auxiliary_grad = torch.autograd.grad(auxiliary, decoded)[0]
        main_norm = float(main_grad.float().norm().item())
        auxiliary_norm = float(auxiliary_grad.float().norm().item())
        if not np.isfinite(main_norm) or not np.isfinite(auxiliary_norm) or main_norm <= 0:
            raise FloatingPointError("non-finite or zero decoder-gradient norm in calibration")
        if auxiliary_norm > 0:
            ratios.append(auxiliary_norm / main_norm)
        component_counts.append(components.count)
        del main_grad, auxiliary_grad, logits, decoded, encoded, images

    if len(ratios) < 6:
        raise ValueError(f"too few nonempty component batches for scale calibration: {len(ratios)}")
    median_ratio = float(np.median(ratios))
    weight = 0.10 / median_ratio
    if not np.isfinite(weight) or not 1e-5 <= weight <= 1.0:
        raise ValueError(f"calibrated coefficient outside locked safety range: {weight}")
    return {
        "training_split_only": True,
        "optimizer_steps": 0,
        "valid_read": False,
        "test_read": False,
        "probe_batches": probe_batches,
        "nonempty_gradient_batches": len(ratios),
        "component_counts_per_batch": component_counts,
        "aux_to_full_main_decoder_gradient_norm_ratios": ratios,
        "median_gradient_norm_ratio": median_ratio,
        "target_auxiliary_gradient_fraction": 0.10,
        "component_weight": weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path,
                        help="model392_cv root containing seed1_fold1..4")
    parser.add_argument("--baseline-root", required=True, type=Path,
                        help="Model392 outputs/model392_widthnorm/seed1")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=1)
    args = parser.parse_args()
    if args.probe_batches < 8:
        raise ValueError("at least 8 batches per fold are preregistered")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {args.out}")
    device = torch.device(args.device)
    folds = {}
    for fold in range(1, 5):
        folds[str(fold)] = calibrate_fold(
            args.split_root / f"seed1_fold{fold}",
            args.baseline_root / f"fold{fold}" / "last.pt",
            device,
            args.probe_batches,
            args.probe_seed,
        )
        print(f"fold={fold} locked_weight={folds[str(fold)]['component_weight']:.12g}", flush=True)
    payload = {
        "experiment": "Model404 Eutectic-Si small/medium component recall loss scale calibration",
        "metric_used_for_calibration": "decoder-feature gradient norm ratio only; no mIoU",
        "formula": "component_weight = 0.10 / median(auxiliary/main decoder-gradient norm ratio)",
        "scope": "fixed four-fold outer-train masks only",
        "valid_read": False,
        "test_read": False,
        "optimizer_steps": 0,
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
