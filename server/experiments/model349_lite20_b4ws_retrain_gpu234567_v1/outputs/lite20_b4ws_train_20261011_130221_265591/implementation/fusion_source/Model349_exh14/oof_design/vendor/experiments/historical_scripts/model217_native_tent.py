#!/usr/bin/env python3
"""Label-free BN-affine TENT adaptation for the model-195 representation branch."""

from __future__ import annotations

import argparse
import copy
import json
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from improve_five_causes import collect_split, decode_mask, scaled_pair, tile_starts
from model181_raw_zcdf_aspp import TeamRepresentationASPP, load_checkpoint, predict_probabilities, representation, score


def corner_inputs(image: np.ndarray, patch_size: int, stride: int, device: torch.device) -> torch.Tensor:
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    feature = representation(scaled)
    height, width = feature.shape[1:]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    locations = list(dict.fromkeys([(ys[0], xs[0]), (ys[0], xs[-1]),
                                    (ys[-1], xs[0]), (ys[-1], xs[-1])]))
    padded_height = max(height, ys[-1] + patch_size)
    padded_width = max(width, xs[-1] + patch_size)
    padded = np.pad(feature, ((0, 0), (0, padded_height - height),
                              (0, padded_width - width)), mode="reflect")
    return torch.stack([torch.from_numpy(padded[:, top:top + patch_size, left:left + patch_size])
                        for top, left in locations]).to(device)


def adapt(checkpoint: str, image: np.ndarray, args, device: torch.device, amp: bool):
    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, checkpoint)
    if args.steps <= 0:
        model.eval()
        return model
    inputs = corner_inputs(image, args.patch_size, args.stride, device)
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True); params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True); params.append(module.bias)
    # The global-pooling BN is unstable when the effective batch is small.
    model.context.pool[2].eval()
    ids = {id(value) for value in params}
    for parameter in model.parameters():
        if id(parameter) not in ids:
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            probability = torch.softmax(model(inputs), dim=1)
            entropy = -(probability * torch.log(probability.clamp_min(1e-7))).sum(1).mean()
        entropy.backward(); optimizer.step()
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    if args.adapt:
        configs = [(0, 0.0)] + [(int(steps), float(lr)) for steps, lr in args.adapt]
    else:
        configs = [(0, 0.0)] + [(steps, lr) for steps in (1, 3, 5) for lr in (5e-6, 1e-5, 2e-5)]
    results = []
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for config_index, (steps, lr) in enumerate(configs):
        predictions, targets, payload = [], [], {}
        for image_path, mask_path in samples:
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            local_args = copy.copy(args); local_args.steps = steps; local_args.lr = lr
            model = adapt(args.checkpoint, image, local_args, device, amp)
            probability = predict_probabilities(model, image_path, args.patch_size, args.stride,
                                                 device, amp, args.tile_batch_size)
            target = decode_mask(Image.open(mask_path).convert("RGB"))
            predictions.append(probability.argmax(axis=0).astype(np.uint8)); targets.append(target)
            payload[image_path.stem] = probability.astype(np.float16)
            print(f"config={config_index} steps={steps} lr={lr} image={image_path.name}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            gc.collect()
        value = score(predictions, targets)
        record = {"config_index": config_index, "steps": steps, "lr": lr, **value}
        results.append(record)
        np.savez_compressed(output / f"valid_probs_{config_index:02d}.npz", **payload)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    best_image = max(results, key=lambda row: row["image_mean_miou"])
    best_exclude = max(results, key=lambda row: row["image_mean_miou_exclude"])
    best_global = max(results, key=lambda row: row["global_miou"])
    result = {"records": results, "best_image": best_image,
              "best_exclude": best_exclude, "best_global": best_global}
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("best_image", "best_exclude", "best_global")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adapt", nargs=2, action="append", metavar=("STEPS", "LR"),
                        help="explicit TENT adaptation config; repeat to evaluate multiple settings")
    main(parser.parse_args())
