#!/usr/bin/env python3
"""Four-fold image-level ensemble for the ResNet34 U-Net++ model 6 family."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from improve_model6 import (
    CLASS_NAMES,
    FineTuneDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    encode_mask,
    evaluate_model,
    segmentation_loss,
    split_parameters,
    tta_logits,
)
from train_models_2_to_6 import UNetPlusPlusResNet34


class SampleDataset(FineTuneDataset):
    def __init__(self, samples, image_size: int, aug_mode: str):
        self.image_paths = [x[0] for x in samples]
        self.mask_paths = [x[1] for x in samples]
        self.image_size = int(image_size)
        self.aug_mode = aug_mode

    @property
    def root(self):
        return Path(".")


def collect_samples(data_dir: str):
    root = Path(data_dir)
    samples = []
    for split in ("train", "valid"):
        for image_path in sorted((root / split / "images").glob("*_image.png")):
            mask_path = root / split / "masks" / image_path.name.replace("_image.png", "_mask.png")
            samples.append((image_path, mask_path))
    return samples


def loaders(train_samples, valid_samples, args):
    train_ds = SampleDataset(train_samples, args.image_size, "safe_plus")
    valid_ds = SampleDataset(valid_samples, args.image_size, "none")
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, persistent_workers=(args.num_workers > 0))
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(valid_ds, shuffle=False, **common))


def optimizer_for(model, args, freeze: bool):
    encoder, decoder = split_parameters(model)
    if freeze:
        for p in encoder:
            p.requires_grad = False
        return torch.optim.AdamW([{"params": decoder, "lr": args.decoder_lr}], weight_decay=args.weight_decay)
    for p in encoder:
        p.requires_grad = True
    return torch.optim.AdamW(
        [{"params": encoder, "lr": args.encoder_lr}, {"params": decoder, "lr": args.decoder_lr}],
        weight_decay=args.weight_decay)


@torch.no_grad()
def test_probabilities(model, data_dir, image_size, device, use_amp):
    model.eval()
    result = []
    images = sorted((Path(data_dir) / "test" / "images").glob("*_image.png"))
    for path in images:
        image = Image.open(path).convert("RGB")
        resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        result.append(tta_logits(model, tensor, use_amp)[0].cpu().numpy())
    return images, np.stack(result)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    all_samples = collect_samples(args.data_dir)
    order = list(range(len(all_samples)))
    random.Random(args.seed).shuffle(order)
    fold_indices = [order[i::4] for i in range(4)]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    probability_sum = None
    test_images = None
    start_all = time.time()

    for fold_id, valid_indices in enumerate(fold_indices, start=1):
        valid_set = set(valid_indices)
        train_samples = [all_samples[i] for i in order if i not in valid_set]
        valid_samples = [all_samples[i] for i in valid_indices]
        print(f"fold={fold_id}/4 train={len(train_samples)} valid={len(valid_samples)}", flush=True)
        train_loader, valid_loader = loaders(train_samples, valid_samples, args)
        model = UNetPlusPlusResNet34(pretrained=True).to(device)
        scaler = GradScaler(enabled=use_amp)
        fold_dir = root / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best = -1.0
        best_epoch = -1
        history = []

        stages = [("freeze", args.freeze_epochs, True), ("finetune", args.finetune_epochs, False)]
        epoch_global = 0
        for stage_name, stage_epochs, freeze in stages:
            optimizer = optimizer_for(model, args, freeze)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1))
            for _ in range(stage_epochs):
                epoch_global += 1
                model.train()
                total = 0.0
                count = 0
                epoch_start = time.time()
                for images, masks in train_loader:
                    images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        loss = segmentation_loss(model(images), masks, "weighted")
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
                    total += float(loss.item()) * images.shape[0]
                    count += images.shape[0]
                valid = evaluate_model(model, valid_loader, device, use_amp, "weighted", use_tta=False)
                scheduler.step()
                record = {"epoch": epoch_global, "stage": stage_name,
                          "train_loss": total / max(count, 1), "valid": valid,
                          "seconds": time.time() - epoch_start}
                history.append(record)
                print(f"fold={fold_id} epoch={epoch_global:02d} stage={stage_name} "
                      f"miou={valid['miou']:.4f} time={record['seconds']:.1f}s", flush=True)
                if float(valid["miou"]) > best:
                    best = float(valid["miou"])
                    best_epoch = epoch_global
                    torch.save({"model": model.state_dict(), "epoch": epoch_global, "metrics": valid}, fold_dir / "best.pt")
        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location="cpu")["model"])
        normal = evaluate_model(model, valid_loader, device, use_amp, "weighted", use_tta=False)
        tta = evaluate_model(model, valid_loader, device, use_amp, "weighted", use_tta=True)
        fold_summary = {"fold": fold_id, "best_epoch": best_epoch, "normal": normal, "tta": tta}
        fold_summaries.append(fold_summary)
        (fold_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
        (fold_dir / "summary.json").write_text(json.dumps(fold_summary, ensure_ascii=False, indent=2))
        test_images, fold_prob = test_probabilities(model, args.data_dir, args.image_size, device, use_amp)
        probability_sum = fold_prob if probability_sum is None else probability_sum + fold_prob
        del model, train_loader, valid_loader
        torch.cuda.empty_cache() if device.type == "cuda" else None

    probability_mean = probability_sum / 4.0
    test_dir = root / "test_masks"
    test_dir.mkdir(parents=True, exist_ok=True)
    for image_path, probability in zip(test_images, probability_mean):
        original_size = Image.open(image_path).size
        prediction = probability.argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).resize(original_size, Image.Resampling.NEAREST).save(
            test_dir / image_path.name.replace("_image.png", "_mask.png"))
    summary = {"folds": fold_summaries,
               "mean_normal_miou": float(np.mean([x["normal"]["miou"] for x in fold_summaries])),
               "mean_tta_miou": float(np.mean([x["tta"]["miou"] for x in fold_summaries])),
               "test_masks": len(test_images), "elapsed_seconds": time.time() - start_all,
               "test_output": str(test_dir)}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
