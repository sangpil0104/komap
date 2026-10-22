#!/usr/bin/env python3
"""Class-balanced reliable-pixel entropy adaptation for Model195."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import (
    TeamRepresentationASPP,
    load_checkpoint,
    predict_probabilities,
    score,
)
from model217_native_tent import corner_inputs


def adapt_reliable_pixels(model: torch.nn.Module, inputs: torch.Tensor,
                          steps: int, lr: float, keep_fraction: float,
                          amp: bool) -> list[dict[str, float]]:
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
    # The global-pooling BN is unstable with only a few corner tiles.
    model.context.pool[2].eval()
    param_ids = {id(parameter) for parameter in params}
    for parameter in model.parameters():
        if id(parameter) not in param_ids:
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(params, lr=lr)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=inputs.device.type, enabled=amp):
            probability = torch.softmax(model(inputs), dim=1)
            entropy = -(probability * probability.clamp_min(1e-7).log()).sum(dim=1)
            pseudo = probability.argmax(dim=1)
            class_losses = []
            selected_counts = []
            for class_id in range(probability.shape[1]):
                class_mask = pseudo == class_id
                values = entropy[class_mask]
                if values.numel() < 16:
                    continue
                cutoff = torch.quantile(values.detach().float(), keep_fraction)
                reliable = values.detach() <= cutoff
                if reliable.any():
                    class_losses.append(values[reliable].mean())
                    selected_counts.append(int(reliable.sum().item()))
            if not class_losses:
                raise RuntimeError("no reliable pseudo-labeled pixels were selected")
            # Equal weight per pseudo-class prevents dominant Al pixels from
            # monopolizing the adaptation gradient.
            loss = torch.stack(class_losses).mean()
        loss.backward()
        optimizer.step()
        history.append({
            "selected_entropy": float(loss.detach()),
            "selected_classes": float(len(class_losses)),
            "selected_pixels": float(sum(selected_counts)),
        })
    model.eval()
    return history


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    if args.limit:
        samples = samples[:args.limit]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_model = TeamRepresentationASPP(pretrained=False)
    load_checkpoint(base_model, args.checkpoint)
    predictions, targets, probability_payload = [], [], {}
    histories: dict[str, list[dict[str, float]]] = {}
    started = time.time()

    for image_index, (image_path, mask_path) in enumerate(samples, start=1):
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        model = copy.deepcopy(base_model).to(device)
        inputs = corner_inputs(image, args.patch_size, args.stride, device)
        histories[image_path.stem] = adapt_reliable_pixels(
            model, inputs, args.steps, args.lr, args.keep_fraction, amp)
        probability = predict_probabilities(
            model, image_path, args.patch_size, args.stride, device, amp,
            args.tile_batch_size)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(probability.argmax(axis=0).astype(np.uint8))
        targets.append(target)
        probability_payload[image_path.stem] = probability.astype(np.float16)
        print(json.dumps({"image_index": image_index, "total": len(samples),
                          "image": image_path.name,
                          "selected_entropy_last": histories[image_path.stem][-1]["selected_entropy"],
                          "selected_classes_last": histories[image_path.stem][-1]["selected_classes"],
                          "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)
        del model, inputs, probability
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = score(predictions, targets)
    np.savez_compressed(output / "valid_probs.npz", **probability_payload)
    result = {
        "model": "Model195 with class-balanced reliable-pixel TENT",
        "adaptation": "per-image BN-affine entropy minimization; for each pseudo-class keep the lowest-entropy fraction and average class losses equally",
        "steps": args.steps,
        "lr": args.lr,
        "keep_fraction_per_pseudo_class": args.keep_fraction,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "split": "valid",
        "metrics": metrics,
        "adaptation_history": histories,
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output / "summary.json"),
                      "image_mean_miou": metrics["image_mean_miou"],
                      "global_miou": metrics["global_miou"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False),
          flush=True)
    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--keep-fraction", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N validation images")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
