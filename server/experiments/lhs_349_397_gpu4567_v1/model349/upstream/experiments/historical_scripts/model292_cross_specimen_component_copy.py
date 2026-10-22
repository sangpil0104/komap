#!/usr/bin/env python3
"""Cross-specimen phase-component copy/paste fine-tuning."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import collect_split, specimen_id
from model181_raw_zcdf_aspp import seed_everything
from model239_native_stem_finetune import Stride1NativeStemASPP
from model253_highres_scale_finetune import (
    HighResolutionPatchDataset,
    dice_loss,
    evaluate,
    freeze_batchnorm_stats,
)


class IdentifiedPatchDataset(HighResolutionPatchDataset):
    def __getitem__(self, index: int):
        features, mask = super().__getitem__(index)
        path, _ = self.samples[index % len(self.samples)]
        return features, mask, specimen_id(path)


def _find_component(mask: np.ndarray, rng: random.Random,
                    minimum_area: int, maximum_fraction: float):
    max_area = int(mask.size * maximum_fraction)
    for class_id in rng.sample((0, 1, 3), 3):
        labels, count = ndimage.label(mask == class_id,
                                      structure=ndimage.generate_binary_structure(2, 1))
        if count == 0:
            continue
        areas = np.bincount(labels.reshape(-1), minlength=count + 1)
        candidates = np.flatnonzero((areas >= minimum_area) & (areas <= max_area))
        candidates = candidates[candidates != 0]
        if len(candidates):
            selected = int(rng.choice(candidates.tolist()))
            component = labels == selected
            ys, xs = np.nonzero(component)
            return class_id, component, (int(ys.min()), int(ys.max()) + 1,
                                         int(xs.min()), int(xs.max()) + 1)
    return None


def cross_specimen_copy_paste(images: torch.Tensor, masks: torch.Tensor,
                              groups, rng: random.Random, probability: float,
                              minimum_area: int, maximum_fraction: float,
                              feather_radius: int) -> int:
    """Paste a translated labeled phase component into the other specimen crop."""
    if images.shape[0] < 2 or len(set(groups)) < 2:
        return 0
    pasted = 0
    for recipient in range(images.shape[0]):
        donor = (recipient + 1) % images.shape[0]
        if groups[recipient] == groups[donor] or rng.random() >= probability:
            continue
        source_mask = masks[donor].numpy()
        found = _find_component(source_mask, rng, minimum_area, maximum_fraction)
        if found is None:
            continue
        class_id, full_component, (y0, y1, x0, x1) = found
        component = full_component[y0:y1, x0:x1]
        height, width = component.shape
        image_height, image_width = source_mask.shape
        if height >= image_height or width >= image_width:
            continue
        top = rng.randrange(image_height - height + 1)
        left = rng.randrange(image_width - width + 1)
        region = torch.from_numpy(component)
        alpha = np.minimum(
            ndimage.distance_transform_edt(component) / max(feather_radius, 1), 1.0
        ).astype(np.float32)
        alpha_tensor = torch.from_numpy(alpha)

        source = images[donor, :, y0:y1, x0:x1]
        destination = images[recipient, :, top:top + height, left:left + width]
        blend = alpha_tensor.unsqueeze(0)
        destination[:, region] = (source[:, region] * blend[0, region]
                                  + destination[:, region] * (1.0 - blend[0, region]))
        target_patch = masks[recipient, top:top + height, left:left + width]
        target_patch[region] = int(class_id)
        pasted += 1
    return pasted


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    encoder, decoder, native = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
        {"params": native, "lr": args.native_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    dataset = IdentifiedPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, 1.0,
        args.seed, global_representation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history, start_time = [], time.time()
    print(f"device={device} train_images={len(train_samples)} valid_images={len(valid_samples)} "
          f"patches={len(dataset)} patch={args.patch_size} copy_paste={args.copy_probability}",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_batchnorm_stats(model)
        total, count, pasted_total = 0.0, 0, 0
        for step, (images, masks, groups) in enumerate(loader):
            rng = random.Random(args.seed + epoch * 1_000_003 + step)
            pasted_total += cross_specimen_copy_paste(
                images, masks, groups, rng, args.copy_probability,
                args.minimum_component_area, args.maximum_component_fraction,
                args.feather_radius)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, masks,
                                       weight=HARD_CLASS_WEIGHTS.to(device)) \
                    + dice_loss(logits, masks)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach()) * images.shape[0]
            count += images.shape[0]

        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / max(count, 1),
                  "copy_pasted_components": pasted_total,
                  "global_miou": valid["global_miou"],
                  "image_mean_miou": valid["image_mean_miou"],
                  "image_mean_miou_exclude": valid["image_mean_miou_exclude"],
                  "global_iou": valid["global_iou"], "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record[args.selection_metric] > best_score:
            best_score, best_state, best_record = (
                float(record[args.selection_metric]), copy.deepcopy(model.state_dict()), record)
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "parent_checkpoint": str(args.checkpoint),
                        "augmentation": "cross-specimen class-component copy/paste"},
                       output / "best.pt")

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **{
        image_path.stem: value.astype(np.float16)
        for (image_path, _), value in zip(valid_samples, probabilities)
    })
    summary = {"method": "cross-specimen component copy/paste fine-tuning",
               "selection_metric": args.selection_metric, "best_selection": best_score,
               "best_record": best_record, "final": final,
               "elapsed_seconds": time.time() - start_time,
               "parent_checkpoint": str(args.checkpoint),
               "copy_paste": {"probability_per_recipient": args.copy_probability,
                              "minimum_component_area": args.minimum_component_area,
                              "maximum_component_fraction": args.maximum_component_fraction,
                              "feather_radius": args.feather_radius}}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "best_selection", "best_record", "final", "elapsed_seconds")},
        ensure_ascii=False), flush=True)
    del loader, dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--native-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--copy-probability", type=float, default=0.40)
    parser.add_argument("--minimum-component-area", type=int, default=96)
    parser.add_argument("--maximum-component-fraction", type=float, default=0.05)
    parser.add_argument("--feather-radius", type=int, default=5)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=292)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
