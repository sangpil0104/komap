#!/usr/bin/env python3
"""Train-only grouped OOF branch for leakage-aware component experiments.

Each fold model sees train specimens outside its held-out fold only. Validation
images are predicted for later evaluation, but validation masks are never read
for training, epoch selection, or fold decisions.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from improve_five_causes import (
    collect_split,
    decode_mask,
    grouped_folds,
    specimen_id,
)
from model181_raw_zcdf_aspp import TeamRepresentationASPP, train_epoch
from model239_native_stem_finetune import Stride1NativeStemASPP
from model288_region_classifier_cv import image_score
from model253_highres_scale_finetune import HighResolutionPatchDataset, predict_scaled


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(device: torch.device) -> Stride1NativeStemASPP:
    """Initialize encoder from ImageNet only; no segmentation checkpoint is used."""
    pretrained = TeamRepresentationASPP(pretrained=True)
    model = Stride1NativeStemASPP()
    incompatible = model.load_state_dict(pretrained.state_dict(), strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith(("native_stem.", "native_to_head."))]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected pretrained initialization mismatch: {missing}, "
                           f"{incompatible.unexpected_keys}")
    del pretrained
    return model.to(device)


def component_metric(probabilities: np.ndarray, mask_path: Path) -> float:
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    return image_score(prediction, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=297)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    train_samples = collect_split(args.data_dir, "train")
    # Keep only paths here. Validation masks are read once after all OOF and
    # validation probabilities have been generated.
    valid_image_paths = sorted((Path(args.data_dir) / "valid" / "images").glob("*_image.png"))
    if not train_samples or not valid_image_paths:
        raise FileNotFoundError("expected non-empty train and valid image splits")
    folds = grouped_folds(train_samples, args.folds, args.seed)
    train_names = [image_path.stem for image_path, _ in train_samples]
    oof_probabilities: dict[str, np.ndarray] = {}
    valid_probability_sum: dict[str, np.ndarray] = {}
    fold_records = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for fold_id, heldout_indices in enumerate(folds, start=1):
        seed_all(args.seed + fold_id)
        heldout_set = set(heldout_indices)
        fold_train = [sample for index, sample in enumerate(train_samples)
                      if index not in heldout_set]
        fold_heldout = [train_samples[index] for index in heldout_indices]
        fold_train_groups = {specimen_id(path) for path, _ in fold_train}
        heldout_groups = {specimen_id(train_samples[index][0]) for index in heldout_indices}
        if fold_train_groups & heldout_groups:
            raise RuntimeError(f"specimen leakage in fold {fold_id}")

        fold_dir = output / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        dataset = HighResolutionPatchDataset(
            fold_train, args.patch_size, args.patches_per_image, 1.0,
            args.seed + fold_id, global_representation=True,
            metallography_augmentation=False, sampling_mode="class_focused")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0))
        model = make_model(device)
        encoder, decoder, native = [], [], []
        for name, parameter in model.named_parameters():
            if name.startswith(("native_stem.", "native_to_head.")):
                native.append(parameter)
            elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
                encoder.append(parameter)
            else:
                decoder.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": encoder, "lr": 3e-5},
            {"params": decoder, "lr": 3e-4},
            {"params": native, "lr": 3e-4},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=use_amp)
        epoch_records = []
        for epoch in _local_progress(range(1, args.epochs + 1), 'model297_train_only_oof_branch'):
            epoch_start = time.time()
            loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
            scheduler.step()
            epoch_record = {"epoch": epoch, "train_loss": loss,
                            "seconds": time.time() - epoch_start}
            epoch_records.append(epoch_record)
            print(f"fold={fold_id}/{args.folds} epoch={epoch}/{args.epochs} "
                  f"train_loss={loss:.4f} seconds={epoch_record['seconds']:.1f}", flush=True)

        torch.save({
            "model": model.state_dict(), "fold": fold_id, "epochs": args.epochs,
            "train_groups": sorted(fold_train_groups),
            "heldout_groups": sorted(heldout_groups),
            "init": "ImageNet ResNet34 only; fresh decoder and native stem",
            "validation_masks_used": False,
        }, fold_dir / "last.pt")
        model.eval()
        heldout_scores = []
        with torch.no_grad():
            for image_path, mask_path in fold_heldout:
                probabilities = predict_scaled(
                    model, image_path, args.patch_size, args.stride, 1.0,
                    device, use_amp, "d4")
                oof_probabilities[image_path.stem] = probabilities.astype(np.float16)
                heldout_scores.append(component_metric(probabilities, mask_path))
            for image_path in valid_image_paths:
                probabilities = predict_scaled(
                    model, image_path, args.patch_size, args.stride, 1.0,
                    device, use_amp, "d4")
                name = image_path.stem
                if name not in valid_probability_sum:
                    valid_probability_sum[name] = probabilities.astype(np.float32)
                else:
                    valid_probability_sum[name] += probabilities.astype(np.float32)

        fold_record = {
            "fold": fold_id, "train_image_count": len(fold_train),
            "heldout_image_count": len(fold_heldout),
            "train_specimen_group_count": len(fold_train_groups),
            "heldout_specimen_groups": sorted(heldout_groups),
            "heldout_train_image_mean_miou": float(np.mean(heldout_scores)),
            "epoch_history": epoch_records,
        }
        fold_records.append(fold_record)
        (output / "progress.json").write_text(
            json.dumps({"completed_folds": fold_records,
                        "completed_oof_images": len(oof_probabilities)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"completed fold={fold_id}: OOF image mIoU="
              f"{fold_record['heldout_train_image_mean_miou']*100:.4f}% "
              f"images={len(oof_probabilities)}/{len(train_samples)}", flush=True)
        del model, loader, dataset, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(oof_probabilities) != set(train_names):
        raise RuntimeError("OOF predictions do not cover every train image exactly once")
    valid_probabilities = {
        name: values / float(args.folds)
        for name, values in valid_probability_sum.items()
    }
    oof_payload = {name: oof_probabilities[name] for name in train_names}
    np.savez_compressed(output / "train_oof_probs.npz", **oof_payload)
    np.savez_compressed(output / "valid_fold_ensemble_probs.npz", **valid_probabilities)

    valid_scores = []
    for image_path in valid_image_paths:
        mask_path = Path(args.data_dir) / "valid" / "masks" / image_path.name.replace(
            "_image.png", "_mask.png")
        valid_scores.append(component_metric(valid_probabilities[image_path.stem], mask_path))
    oof_image_count = sum(fold["heldout_image_count"] for fold in fold_records)
    train_oof_image_mean = sum(
        fold["heldout_train_image_mean_miou"] * fold["heldout_image_count"]
        for fold in fold_records
    ) / max(oof_image_count, 1)
    result = {
        "model": 297,
        "method": "train-only specimen-group OOF ResNet34 U-Net++/ASPP branch",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "train_image_count": len(train_samples),
        "train_specimen_group_count": len({specimen_id(path) for path, _ in train_samples}),
        "valid_image_count": len(valid_image_paths),
        "fold_count": args.folds,
        "fold_epochs": args.epochs,
        "validation_masks_used_for_training_or_selection": False,
        "folds": fold_records,
        "train_oof_image_mean_miou": float(train_oof_image_mean),
        "valid_fold_ensemble_image_mean_miou": float(np.mean(valid_scores)),
        "valid_image_scores": valid_scores,
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            "train_oof_probs": str(output / "train_oof_probs.npz"),
            "valid_fold_ensemble_probs": str(output / "valid_fold_ensemble_probs.npz"),
        },
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "train_oof_image_mean_miou", "valid_fold_ensemble_image_mean_miou",
        "fold_count", "elapsed_seconds", "artifacts")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
