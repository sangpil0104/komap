#!/usr/bin/env python3
"""Per-image marginal-entropy TENT across D4 views, with memory-bounded replay."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import inverse_view, transform_view, views
from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import (
    TeamRepresentationASPP,
    load_checkpoint,
    predict_probabilities,
    score,
)
from model217_native_tent import corner_inputs


def adapt(model: torch.nn.Module, inputs: torch.Tensor, steps: int, lr: float,
          amp: bool) -> list[float]:
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
            if module is not model.context.pool[2]:
                module.momentum = None
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.zero_()
    model.context.pool[2].eval()
    ids = {id(value) for value in params}
    for parameter in model.parameters():
        if id(parameter) not in ids:
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)
    d4_views = tuple(views("d4"))
    history = []
    batch, _, height, width = inputs.shape
    for _ in range(steps):
        # First pass estimates the marginal prediction without retaining 8 graphs.
        with torch.no_grad():
            probability_sum = torch.zeros(
                (batch, 4, height, width), device=inputs.device, dtype=torch.float32)
            for rotation, flip in d4_views:
                augmented = transform_view(inputs, rotation, flip)
                with torch.autocast(device_type="cuda", enabled=amp):
                    probability = torch.softmax(model(augmented), dim=1)
                probability_sum += inverse_view(probability, rotation, flip).float()
            marginal = probability_sum / len(d4_views)
            entropy = -(marginal * torch.log(marginal.clamp_min(1e-7))).sum(1).mean()
            grad_marginal = -(torch.log(marginal.clamp_min(1e-7)) + 1.0)
            grad_marginal /= (batch * height * width)
        optimizer.zero_grad(set_to_none=True)
        # Replay one view at a time; the analytical marginal-entropy gradient
        # lets each activation graph be freed immediately after backward.
        for rotation, flip in d4_views:
            augmented = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=amp):
                probability = torch.softmax(model(augmented), dim=1)
            probability = inverse_view(probability, rotation, flip)
            torch.autograd.backward(
                probability,
                grad_tensors=(grad_marginal / len(d4_views)).to(probability.dtype),
            )
        optimizer.step()
        history.append(float(entropy))
    return history


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = TeamRepresentationASPP(pretrained=False)
    load_checkpoint(base, args.checkpoint)
    predictions, targets, payload = [], [], {}
    histories = {}
    for image_path, mask_path in samples:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        inputs = corner_inputs(image, args.patch_size, args.stride, device)
        model = copy.deepcopy(base).to(device)
        histories[image_path.stem] = adapt(model, inputs, args.steps, args.lr, amp)
        model.eval()
        probability = predict_probabilities(
            model, image_path, args.patch_size, args.stride, device, amp,
            args.tile_batch_size)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(probability.argmax(axis=0).astype(np.uint8))
        targets.append(target)
        payload[image_path.stem] = probability.astype(np.float16)
        print(json.dumps({"image": image_path.name,
                          "marginal_entropy": histories[image_path.stem]},
                         ensure_ascii=False), flush=True)
        del model, inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = score(predictions, targets)
    np.savez_compressed(output / "valid_probs.npz", **payload)
    result = {"adaptation": "per-image D4 marginal-entropy TENT with replayed gradients",
              "steps": args.steps, "lr": args.lr,
              "mean_entropy_history": {
                  str(step): float(np.mean([value[step] for value in histories.values()]))
                  for step in range(args.steps)},
              **metrics}
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del base
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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    main(parser.parse_args())
