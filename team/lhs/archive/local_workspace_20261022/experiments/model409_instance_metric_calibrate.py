#!/usr/bin/env python3
"""Outer-train-only, zero-step gradient calibration for Model409."""

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
sys.path.insert(0, str(ROOT / "model336_hrnet_widthnorm_candidate"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from komap_seg.checkpoint import load_model_and_preprocessing  # noqa: E402
from komap_seg.dataset import KomapPatchDataset, modal_image_width, seed_worker  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from model409_instance_metric_loss_v2 import (  # noqa: E402
    ProjectionHead,
    build_component_pairs,
    component_local_metric_loss,
)


def calibrate_fold(
    data_root: Path,
    checkpoint: Path,
    device: torch.device,
    probe_examples: int,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, preprocessing = load_model_and_preprocessing(checkpoint, device)
    target_width = modal_image_width(data_root / "train/images")
    if target_width != preprocessing["target_width"]:
        raise ValueError("outer-train modal width differs from parent checkpoint")
    dataset = KomapPatchDataset(
        data_root / "train/images",
        data_root / "train/masks",
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
        batch_size=1,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    channels = model.segmentation_head[0].in_channels
    cuda_devices = [] if device.type != "cuda" else [device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed + 409_000)
        projector = ProjectionHead(channels).to(device)
    segmentation_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    ratios: list[float] = []
    component_counts: list[int] = []
    cell_support = {f"class{class_id}_bin{bin_index}": 0
                    for class_id in (1, 3) for bin_index in range(3)}
    model.train()
    projector.train()
    for example_index, (images_cpu, masks_cpu) in enumerate(loader):
        if example_index >= probe_examples:
            break
        targets = build_component_pairs(masks_cpu)
        component_counts.append(targets.count)
        for pair in targets.pairs:
            cell_support[f"class{pair.class_id}_bin{pair.bin_index}"] += 1
        if targets.count == 0:
            continue
        images = images_cpu.to(device, non_blocking=True)
        masks = masks_cpu.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoded = model.encoder(images)
            decoded = model.decoder(encoded)
            logits = model.segmentation_head(decoded)
            boundary_logits = model.boundary_head(decoded)
            projected = projector(decoded)
            main = segmentation_loss(logits, masks) + boundary_loss(boundary_logits, masks)
            auxiliary = component_local_metric_loss(projected, targets)
        if not torch.isfinite(auxiliary.detach()) or auxiliary.detach().item() <= 0:
            del logits, boundary_logits, projected, decoded, encoded, images, masks, main, auxiliary
            continue
        main_grad = torch.autograd.grad(main, decoded, retain_graph=True)[0]
        auxiliary_grad = torch.autograd.grad(auxiliary, decoded)[0]
        main_norm = float(main_grad.float().norm().item())
        auxiliary_norm = float(auxiliary_grad.float().norm().item())
        if (not np.isfinite(main_norm) or not np.isfinite(auxiliary_norm)
                or main_norm <= 0 or auxiliary_norm <= 0):
            raise FloatingPointError("non-finite or zero decoder-gradient norm")
        ratios.append(auxiliary_norm / main_norm)
        del main_grad, auxiliary_grad, logits, boundary_logits, projected, decoded, encoded
        del images, masks, main, auxiliary

    minimum = int(np.ceil(0.75 * probe_examples))
    if len(ratios) < minimum:
        raise ValueError(f"insufficient active local-pair crops: {len(ratios)} < {minimum}")
    median_ratio = float(np.median(ratios))
    weight = 0.10 / median_ratio
    if not np.isfinite(weight) or not 1e-5 <= weight <= 1.0:
        raise ValueError(f"unsafe calibrated metric coefficient: {weight}")
    return {
        "training_split_only": True,
        "optimizer_steps": 0,
        "valid_read": False,
        "test_read": False,
        "probe_examples": probe_examples,
        "probe_batch_size": 1,
        "active_pair_crops": len(ratios),
        "component_pairs_per_crop": component_counts,
        "component_pairs_by_class_and_size_bin": cell_support,
        "active_aux_to_main_decoder_gradient_norm_ratios": ratios,
        "median_active_gradient_norm_ratio": median_ratio,
        "target_auxiliary_gradient_fraction": 0.10,
        "metric_weight": weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=409)
    args = parser.parse_args()
    if args.probe_examples != 32:
        raise ValueError("Model409 fixes 32 outer-train probe crops/fold")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.out}")
    folds = {}
    for fold in range(1, 5):
        row = calibrate_fold(
            args.split_root / f"seed1_fold{fold}",
            args.baseline_root / f"fold{fold}/last.pt",
            torch.device(args.device),
            args.probe_examples,
            args.seed,
        )
        folds[str(fold)] = row
        print(
            f"fold={fold} metric_weight={row['metric_weight']:.12g} "
            f"active_pairs={row['active_pair_crops']}/32 "
            f"components={sum(row['component_pairs_per_crop'])} "
            f"median_grad_ratio={row['median_active_gradient_norm_ratio']:.6g}",
            flush=True,
        )
    if any(
        item["optimizer_steps"] != 0 or item["valid_read"] or item["test_read"]
        or item["active_pair_crops"] < 24
        for item in folds.values()
    ):
        raise ValueError("Model409 outer-train-only calibration safety gate failed")
    result = {
        "experiment": "Model409 v2 tiny-component local feature metric calibration",
        "objective": "same-instance target positives vs 2-6px local fixed-confuser negatives",
        "fixed_confusers": {"Al3Ni": ["Al"], "Eutectic Si": ["Al", "Al3Ni"]},
        "metric_used_for_calibration": "median active per-crop decoder-gradient norm ratio",
        "probe_examples_per_fold": 32,
        "probe_batch_size": 1,
        "scope": "four fixed outer-train folds only",
        "valid_read": False,
        "test_read": False,
        "optimizer_steps": 0,
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
