#!/usr/bin/env python3
"""Unlabeled transductive TENT: adapt shared BN affine values over valid images."""
from __future__ import annotations

import argparse
import gc
import json
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


def configure_bn_tent(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
    # Avoid unstable running-stat updates for the 1x1 image-pooling BN.
    model.context.pool[2].eval()
    ids = {id(value) for value in params}
    for parameter in model.parameters():
        if id(parameter) not in ids:
            parameter.requires_grad_(False)
    return params


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Cache only unlabeled input tiles; labels are not read until scoring.
    tiles = []
    for image_path, _ in samples:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tiles.append(corner_inputs(image, args.patch_size, args.stride, device))
        print(f"cached adaptation input {image_path.name}", flush=True)

    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint)
    params = configure_bn_tent(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total_entropy = 0.0
        for index, inputs in enumerate(tiles):
            with torch.autocast(device_type="cuda", enabled=amp):
                probability = torch.softmax(model(inputs), dim=1)
                entropy = -(probability * torch.log(probability.clamp_min(1e-7))).sum(1).mean()
            (entropy / len(tiles)).backward()
            total_entropy += float(entropy.detach()) / len(tiles)
        optimizer.step()
        history.append(total_entropy)
        print(json.dumps({"global_step": step + 1, "mean_entropy": total_entropy,
                          "lr": args.lr}, ensure_ascii=False), flush=True)

    model.eval()
    state_path = output / "adapted_state.pt"
    torch.save({"model": model.state_dict(), "steps": args.steps,
                "lr": args.lr, "mean_entropy_history": history}, state_path)

    predictions, targets, payload = [], [], {}
    for image_path, mask_path in samples:
        probability = predict_probabilities(
            model, image_path, args.patch_size, args.stride, device, amp,
            args.tile_batch_size)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(probability.argmax(axis=0).astype(np.uint8))
        targets.append(target)
        payload[image_path.stem] = probability.astype(np.float16)
        print(f"predicted {image_path.name}", flush=True)
    metrics = score(predictions, targets)
    np.savez_compressed(output / "valid_probs.npz", **payload)
    result = {"adaptation": "shared unlabeled BN-affine TENT over all valid images",
              "steps": args.steps, "lr": args.lr,
              "mean_entropy_history": history, **metrics}
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del model, tiles
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
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    main(parser.parse_args())
