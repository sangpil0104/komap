#!/usr/bin/env python3
"""Outer-train-only zero-step gradient calibration for Model408."""

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
from model408_tiny_confusion_loss import (  # noqa: E402
    build_component_targets,
    tiny_confusion_hinge_loss,
)


def calibrate_fold(
    data_root: Path,
    checkpoint: Path,
    device: torch.device,
    probe_examples: int,
    probe_seed: int,
) -> dict:
    random.seed(probe_seed)
    np.random.seed(probe_seed)
    torch.manual_seed(probe_seed)
    model, preprocessing = load_model_and_preprocessing(checkpoint, device)
    target_width = modal_image_width(data_root / "train/images")
    if target_width != preprocessing["target_width"]:
        raise ValueError(f"train mode width differs from checkpoint: {target_width}")
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
    segmentation_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    ratios: list[float] = []
    component_counts: list[int] = []
    hinge_active_examples = 0
    model.train()
    for example_index, (images_cpu, masks_cpu) in enumerate(loader):
        if example_index >= probe_examples:
            break
        targets = build_component_targets(masks_cpu)
        component_counts.append(targets.count)
        if targets.count == 0:
            continue
        images = images_cpu.to(device, non_blocking=True)
        masks = masks_cpu.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoded = model.encoder(images)
            decoded = model.decoder(encoded)
            logits = model.segmentation_head(decoded)
            boundary_logits = model.boundary_head(decoded)
            main = segmentation_loss(logits, masks) + boundary_loss(boundary_logits, masks)
            auxiliary = tiny_confusion_hinge_loss(logits, targets)
        if float(auxiliary.detach().float().item()) <= 0.0:
            del logits, boundary_logits, decoded, encoded, images, masks, main, auxiliary
            continue
        main_grad = torch.autograd.grad(main, decoded, retain_graph=True)[0]
        auxiliary_grad = torch.autograd.grad(auxiliary, decoded)[0]
        main_norm = float(main_grad.float().norm().item())
        auxiliary_norm = float(auxiliary_grad.float().norm().item())
        if (not np.isfinite(main_norm) or not np.isfinite(auxiliary_norm)
                or main_norm <= 0 or auxiliary_norm <= 0):
            raise FloatingPointError("non-finite or zero decoder-gradient norm in calibration")
        ratios.append(auxiliary_norm / main_norm)
        hinge_active_examples += 1
        del main_grad, auxiliary_grad, logits, boundary_logits, decoded, encoded
        del images, masks, main, auxiliary

    minimum = int(np.ceil(0.75 * probe_examples))
    if len(ratios) < minimum:
        raise ValueError(
            f"too few active confusion-hinge examples: {len(ratios)} < {minimum}"
        )
    median_ratio = float(np.median(ratios))
    weight = 0.10 / median_ratio
    if not np.isfinite(weight) or not 1e-5 <= weight <= 1.0:
        raise ValueError(f"calibrated coefficient outside locked safety range: {weight}")
    return {
        "training_split_only": True,
        "optimizer_steps": 0,
        "valid_read": False,
        "test_read": False,
        "probe_examples": probe_examples,
        "probe_batch_size": 1,
        "active_hinge_examples": hinge_active_examples,
        "gt_component_counts_per_example": component_counts,
        "active_aux_to_main_decoder_gradient_norm_ratios": ratios,
        "median_active_gradient_norm_ratio": median_ratio,
        "target_auxiliary_gradient_fraction": 0.10,
        "component_weight": weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-examples", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=408)
    args = parser.parse_args()
    if args.probe_examples != 32:
        raise ValueError("Model408 preregisters exactly 32 outer-train probe crops per fold")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.out}")

    folds = {}
    for fold in range(1, 5):
        record = calibrate_fold(
            args.split_root / f"seed1_fold{fold}",
            args.baseline_root / f"fold{fold}/last.pt",
            torch.device(args.device),
            args.probe_examples,
            args.probe_seed,
        )
        folds[str(fold)] = record
        print(
            f"fold={fold} component_weight={record['component_weight']:.12g} "
            f"median_active_grad_ratio={record['median_active_gradient_norm_ratio']:.6g} "
            f"active_hinge={record['active_hinge_examples']}/32",
            flush=True,
        )
    if any(
        not item["training_split_only"] or item["optimizer_steps"] != 0
        or item["valid_read"] or item["test_read"]
        or item["active_hinge_examples"] < 24
        or not 1e-5 <= item["component_weight"] <= 1.0
        for item in folds.values()
    ):
        raise ValueError("outer-train calibration failed its locked safety checks")

    result = {
        "experiment": "Model408 tiny-GT confusion hinge calibration",
        "objective": (
            "component-normalized zero-margin hinge on currently confused pixels "
            "of augmented-grid 1-64px-proxy Al3Ni/Eutectic-Si GT components"
        ),
        "fixed_confusers": {"Al3Ni": ["Al"], "Eutectic Si": ["Al", "Al3Ni"]},
        "metric_used_for_calibration": "median active per-crop decoder-gradient norm ratio",
        "formula": (
            "component_weight = 0.10 / median(norm(grad(confusion hinge)) / "
            "norm(grad(CE+Dice+boundary)))"
        ),
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
