#!/usr/bin/env python3
"""Outer-train-only, zero-step gradient calibration for Model411."""

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
sys.path.insert(0, str(ROOT / "experiments"))

from komap_seg.checkpoint import load_model_and_preprocessing  # noqa: E402
from komap_seg.dataset import KomapPatchDataset, modal_image_width, seed_worker  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from model411_bicc_loss import (  # noqa: E402
    GT_BRANCH_WEIGHT,
    PRED_BRANCH_WEIGHT,
    build_component_targets,
    build_unmatched_predicted_components,
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
    gt_counts: list[int] = []
    pred_counts: list[int] = []
    model.train()
    for example_index, (images_cpu, masks_cpu) in enumerate(loader):
        if example_index >= probe_examples:
            break
        component_targets = build_component_targets(masks_cpu)
        images = images_cpu.to(device, non_blocking=True)
        masks = masks_cpu.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoded = model.encoder(images)
            decoded = model.decoder(encoded)
            logits = model.segmentation_head(decoded)
            boundary_logits = model.boundary_head(decoded)
            main = seg_loss(logits, masks) + boundary_loss(boundary_logits, masks)
            auxiliary = component_recall_loss(logits, component_targets)
            predicted = build_unmatched_predicted_components(logits, masks_cpu)

        main_grad = torch.autograd.grad(main, decoded, retain_graph=True)[0]
        auxiliary_grad = torch.autograd.grad(auxiliary, decoded)[0]
        main_norm = float(main_grad.float().norm().item())
        auxiliary_norm = float(auxiliary_grad.float().norm().item())
        if not np.isfinite(main_norm) or not np.isfinite(auxiliary_norm) or main_norm <= 0:
            raise FloatingPointError("non-finite or zero decoder-gradient norm")
        if auxiliary_norm > 0:
            ratios.append(auxiliary_norm / main_norm)
        gt_counts.append(component_targets.gt_components.count)
        pred_counts.append(predicted.count)
        del main_grad, auxiliary_grad, logits, boundary_logits, decoded, encoded
        del images, masks, main, auxiliary, predicted

    minimum = int(np.ceil(0.75 * probe_examples))
    if len(ratios) < minimum:
        raise ValueError(f"too few active auxiliary probes: {len(ratios)} < {minimum}")
    median_ratio = float(np.median(ratios))
    coefficient = 0.10 / median_ratio
    if not np.isfinite(coefficient) or not 1e-5 <= coefficient <= 1.0:
        raise ValueError(f"calibrated coefficient outside fixed safety range: {coefficient}")
    return {
        "training_split_only": True,
        "optimizer_steps": 0,
        "valid_read": False,
        "test_read": False,
        "probe_examples": probe_examples,
        "active_aux_gradient_examples": len(ratios),
        "gt_component_counts": gt_counts,
        "unmatched_predicted_component_counts": pred_counts,
        "aux_to_main_decoder_gradient_norm_ratios": ratios,
        "median_aux_to_main_ratio": median_ratio,
        "target_auxiliary_gradient_fraction": 0.10,
        "coefficient": coefficient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-examples", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=411)
    args = parser.parse_args()
    if args.probe_examples != 32:
        raise ValueError("Model411 fixes exactly 32 outer-train probes per fold")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {args.out}")

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
            f"fold={fold} coefficient={record['coefficient']:.12g} "
            f"active={record['active_aux_gradient_examples']}/32",
            flush=True,
        )
    if any(
        item["optimizer_steps"] != 0
        or item["valid_read"]
        or item["test_read"]
        or item["active_aux_gradient_examples"] < 24
        or not 1e-5 <= item["coefficient"] <= 1.0
        for item in folds.values()
    ):
        raise ValueError("Model411 outer-train calibration failed its fixed safety gate")

    result = {
        "experiment": "Model411 component-balanced bidirectional small-island loss",
        "objective": "0.75 GT component recall + 0.25 mean-reduced unmatched predicted-component penalty",
        "metric_used_for_calibration": "decoder-gradient norm ratio only; no mIoU",
        "formula": "coefficient = 0.10 / median(auxiliary/main decoder-gradient norm ratio)",
        "scope": "four fixed outer-train folds only; no held-out/official valid/test",
        "fixed_branch_weights": {
            "ground_truth_components": GT_BRANCH_WEIGHT,
            "unmatched_predicted_components": PRED_BRANCH_WEIGHT,
        },
        "optimizer_steps": 0,
        "valid_read": False,
        "test_read": False,
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
