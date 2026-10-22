#!/usr/bin/env python3
"""Low-memory, outer-train-only Model406 loss-scale calibration.

Processes the same 32 sampled crops per fold as the v1 calibration in
microbatches of one, so the calibration's model activation peak is reduced.
The calibration statistic is explicitly the median per-crop decoder-gradient
ratio; it is not presented as a batch-of-four estimate.
"""

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
from model406_component_hardnegative_loss import (  # noqa: E402
    build_component_targets,
    component_recall_loss,
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
        batch_size=1,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    seg_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    ratios: list[float] = []
    component_counts: list[int] = []
    model.train()
    for example_index, (images_cpu, masks_cpu) in enumerate(loader):
        if example_index >= probe_examples:
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
        del main_grad, auxiliary_grad, logits, boundary_logits, decoded, encoded
        del images, masks, main, auxiliary

    minimum = int(np.ceil(0.75 * probe_examples))
    if len(ratios) < minimum:
        raise ValueError(
            f"too few nonempty component examples for calibration: {len(ratios)} < {minimum}"
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
        "nonempty_gradient_examples": len(ratios),
        "component_counts_per_example": component_counts,
        "joint_aux_to_main_decoder_gradient_norm_ratios": ratios,
        "median_joint_gradient_norm_ratio": median_ratio,
        "target_joint_auxiliary_gradient_fraction": 0.10,
        "joint_aux_weight": weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-examples", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=406)
    args = parser.parse_args()
    if args.probe_examples != 32:
        raise ValueError("v2 preregisters exactly 32 outer-train probe crops per fold")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.out}")
    folds = {}
    for fold in range(1, 5):
        record = calibrate_fold(
            args.split_root / f"seed1_fold{fold}",
            args.baseline_root / f"fold{fold}" / "last.pt",
            torch.device(args.device),
            args.probe_examples,
            args.probe_seed,
        )
        folds[str(fold)] = record
        print(
            f"fold={fold} joint_aux_weight={record['joint_aux_weight']:.12g} "
            f"median_per_crop_grad_ratio={record['median_joint_gradient_norm_ratio']:.6g} "
            f"nonempty={record['nonempty_gradient_examples']}/32",
            flush=True,
        )
    if any(
        not item["training_split_only"] or item["optimizer_steps"] != 0
        or item["valid_read"] or item["test_read"]
        or item["nonempty_gradient_examples"] < 24
        or not 1e-5 <= item["joint_aux_weight"] <= 1.0
        for item in folds.values()
    ):
        raise ValueError("outer-train calibration failed its locked safety checks")
    result = {
        "experiment": "Model406 v2 component-recall plus hard-negative calibration",
        "objective": (
            "component_recall_loss + hard_negative_island_loss; top 0.1% negative "
            "probabilities outside a one-pixel same-class GT halo"
        ),
        "metric_used_for_calibration": (
            "median per-crop decoder-feature gradient norm ratio; no mIoU or held-out images"
        ),
        "formula": (
            "joint_aux_weight = 0.10 / median(norm(grad(joint_aux))/"
            "norm(grad(CE+Dice+boundary)))"
        ),
        "probe_batch_size": 1,
        "probe_examples_per_fold": 32,
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
