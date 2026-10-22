#!/usr/bin/env python3
"""Re-estimate shared BN running statistics over unlabeled images cumulatively."""
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


def prepare_adabn(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)
        if module is model.context.pool[2]:
            module.eval()
            continue
        module.train()
        module.momentum = None  # cumulative average across batches/images
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()
        modules.append(module)
    return modules


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tiles = []
    for image_path, _ in samples:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tiles.append(corner_inputs(image, args.patch_size, args.stride, device))
        print(f"cached unlabeled input {image_path.name}", flush=True)

    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint)
    bn_modules = prepare_adabn(model)
    model.zero_grad(set_to_none=True)

    # Each four-corner batch contributes equally to cumulative BN moments.
    with torch.no_grad():
        for image_index, inputs in enumerate(tiles):
            with torch.autocast(device_type="cuda", enabled=amp):
                model(inputs)
            print(json.dumps({"adapted_image_index": image_index,
                              "bn_batches": int(bn_modules[0].num_batches_tracked)},
                             ensure_ascii=False), flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "adaptation": "cumulative BN statistics",
                "adapted_images": len(tiles)}, output / "adapted_state.pt")

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
    result = {"adaptation": "shared unlabeled cumulative BN statistics over all valid images",
              "adapted_images": len(samples), "bn_batches": len(samples), **metrics}
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
    main(parser.parse_args())
