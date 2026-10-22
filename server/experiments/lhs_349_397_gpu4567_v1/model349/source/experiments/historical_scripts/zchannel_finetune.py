#!/usr/bin/env python3
"""Fine-tune U-Net++ with an additional per-image z-score grayscale channel."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import (IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
                                  RareAwarePatchDataset, collect_split, decode_mask,
                                  encode_mask, scaled_pair, tile_starts)
from train_models_2_to_6 import UNetPlusPlusResNet34


class ZDataset(RareAwarePatchDataset):
    def __getitem__(self, index):
        rgb, mask = super().__getitem__(index)
        raw = rgb * torch.as_tensor(IMAGENET_STD, dtype=rgb.dtype)[:, None, None]
        raw = raw + torch.as_tensor(IMAGENET_MEAN, dtype=rgb.dtype)[:, None, None]
        gray = raw.mean(dim=0)
        z = (gray - gray.mean()) / gray.std().clamp_min(1e-5)
        return torch.cat([rgb, z.unsqueeze(0)], dim=0), mask


class ZUNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = UNetPlusPlusResNet34(pretrained=False)
        old = self.base.stem[0]
        self.base.stem[0] = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                                      old.padding, bias=False)

    def load_checkpoint(self, checkpoint: str):
        payload = torch.load(checkpoint, map_location="cpu")
        source = payload["model"]
        target = self.base.state_dict()
        compatible = {key: value for key, value in source.items()
                      if key in target and target[key].shape == value.shape}
        self.base.load_state_dict(compatible, strict=False)
        old_weight = source["stem.0.weight"]
        with torch.no_grad():
            self.base.stem[0].weight[:, :3].copy_(old_weight)
            self.base.stem[0].weight[:, 3:4].copy_(old_weight.mean(dim=1, keepdim=True))

    def forward(self, x):
        return self.base(x)


def normalize_z(patch: np.ndarray) -> torch.Tensor:
    raw = patch.astype(np.float32) / 255.0
    rgb = (raw - IMAGENET_MEAN) / IMAGENET_STD
    gray = raw.mean(axis=2)
    z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    return torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), z[None]], axis=0)).float()


@torch.no_grad()
def predict_z(model, image_path, patch_size, stride, device, use_amp, tile_batch_size):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([normalize_z(padded[top:top + patch_size, left:left + patch_size])
                              for top, left in batch_locations]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in [(0, False), (0, True), (1, False), (1, True)]:
            view = torch.rot90(inputs, rotation, dims=(-2, -1))
            if flip:
                view = torch.flip(view, dims=(-1,))
            with autocast(enabled=use_amp):
                probabilities = torch.softmax(model(view), dim=1)
            if flip:
                probabilities = torch.flip(probabilities, dims=(-1,))
            probabilities = torch.rot90(probabilities, -rotation, dims=(-2, -1))
            probability_batch = probabilities if probability_batch is None else probability_batch + probabilities
        probability_batch = (probability_batch / 4.0).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index]
            weight_sum[top:top + patch_size, left:left + patch_size] += 1.0
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([np.asarray(Image.fromarray(probabilities[class_id]).resize(
        original_size, Image.Resampling.BILINEAR), dtype=np.float32) for class_id in range(NUM_CLASSES)])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


def dice_loss(logits, target):
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = ZDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = ZUNetPlusPlus().to(device)
    model.load_checkpoint(args.checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best = -1.0; history=[]; start_time=time.time()
    for epoch in _local_progress(range(1, args.epochs + 1), 'zchannel_finetune'):
        model.train(); total=0.0; count=0; epoch_start=time.time()
        for images, masks in _local_progress(loader, 'zchannel_finetune'):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, masks, weight=HARD_CLASS_WEIGHTS.to(device)) + dice_loss(logits, masks)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * images.shape[0]; count += images.shape[0]
        model.eval(); confusion=np.zeros((4,4),dtype=np.int64); image_scores=[]
        for image_path, mask_path in _local_progress(valid_samples, 'zchannel_finetune'):
            probability=predict_z(model,image_path,args.patch_size,args.val_stride,device,use_amp,args.tile_batch_size)
            prediction=probability.argmax(0).astype(np.uint8); target=decode_mask(Image.open(mask_path).convert("RGB"))
            bins=target.reshape(-1)*4+prediction.reshape(-1); current=np.bincount(bins,minlength=16).reshape(4,4); confusion+=current
            inter=np.diag(current).astype(np.float64); union=current.sum(0)+current.sum(1)-inter
            image_scores.append(float(np.mean(inter/np.maximum(union,1.0))))
        inter=np.diag(confusion).astype(np.float64); union=confusion.sum(0)+confusion.sum(1)-inter; iou=inter/np.maximum(union,1.0)
        valid={"image_miou_compat":float(iou.mean()),"image_mean_miou":float(np.mean(image_scores)),"image_class_iou":iou.tolist()}
        history.append({"epoch":epoch,"train_loss":total/max(count,1),"valid":valid,"seconds":time.time()-epoch_start})
        (output/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
        torch.save({"model":model.state_dict(),"epoch":epoch,"metrics":valid,"args":vars(args)},output/"last.pt")
        print(f"epoch={epoch:02d} loss={total/max(count,1):.4f} image={valid['image_mean_miou']:.4f} image_compat={valid['image_miou_compat']:.4f} iou="+",".join(f"{v:.3f}" for v in iou),flush=True)
        if valid["image_mean_miou"]>best:
            best=valid["image_mean_miou"]; torch.save({"model":model.state_dict(),"epoch":epoch,"metrics":valid,"args":vars(args)},output/"best.pt")
    summary={"best_image_mean_miou":best,"epochs":len(history),"elapsed_seconds":time.time()-start_time}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output-dir",required=True)
    parser.add_argument("--patch-size",type=int,default=640); parser.add_argument("--val-stride",type=int,default=512); parser.add_argument("--tile-batch-size",type=int,default=1); parser.add_argument("--patches-per-image",type=int,default=4); parser.add_argument("--epochs",type=int,default=4); parser.add_argument("--batch-size",type=int,default=1); parser.add_argument("--num-workers",type=int,default=4); parser.add_argument("--lr",type=float,default=1e-5); parser.add_argument("--weight-decay",type=float,default=1e-4); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--no-amp",action="store_true")
    main(parser.parse_args())
