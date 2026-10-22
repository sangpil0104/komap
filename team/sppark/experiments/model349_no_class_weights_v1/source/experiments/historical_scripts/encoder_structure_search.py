#!/usr/bin/env python3
"""Native-patch encoder/decoder structure search for KoMaP.

The structure candidates are intentionally evaluated under the same protocol
as model98/99/101:
  * aspect-preserving 512px patches;
  * rare-class aware crop sampling;
  * unweighted CE + Dice;
  * D4 TTA and Gaussian overlap blending for validation/test.

Candidates:
  aspp       ResNet34 U-Net++ + residual ASPP bottleneck
  highres    U-Net++ + residual high-resolution early-feature branch
  gated      U-Net++ + gated 1x1 adapters on encoder stages
  bnfreeze   model98 architecture with frozen encoder BN running statistics
  bngn       encoder BN freeze plus GroupNorm in the decoder
  convnext   ConvNeXt-Tiny U-Net under the native-patch protocol
  hrnet      HRNet-W18 multi-resolution FPN under the native-patch protocol
  effb4      EfficientNet-B4 U-Net under the native-patch protocol

Structure candidates can be initialized from model98. Diverse backbones start
from their ImageNet weights and are deliberately not compared to the old
square-resize runs.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from advanced_inference import inverse_view, predict_probabilities, transform_view, views
from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import (
    CLASS_NAMES,
    NUM_CLASSES,
    RareAwarePatchDataset,
    collect_split,
    decode_mask,
    encode_mask,
    normalize_image,
)
from train_models_2_to_6 import ConvNeXtTinyUNet, UNetPlusPlusResNet34
from train_models_72_74 import HRNetW18FPN


STRUCTURE_CANDIDATES = ("aspp", "highres", "gated", "bnfreeze", "bngn")
BACKBONE_CANDIDATES = ("convnext", "hrnet", "effb4", "resnet50", "resnet50pp")
ALL_CANDIDATES = STRUCTURE_CANDIDATES + BACKBONE_CANDIDATES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualASPP(nn.Module):
    """ASPP whose residual scale starts at zero for checkpoint compatibility."""

    def __init__(self, channels: int = 512, branch_channels: int = 128) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            self._branch(channels, branch_channels, 1),
            self._branch(channels, branch_channels, 3, dilation=6),
            self._branch(channels, branch_channels, 3, dilation=12),
            self._branch(channels, branch_channels, 3, dilation=18),
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _branch(channels: int, branch_channels: int, kernel: int, dilation: int = 1) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel, padding=dilation * (kernel // 2),
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        values = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        values.append(pooled)
        return x + self.scale * self.project(torch.cat(values, dim=1))


class ASPPUNetPP(UNetPlusPlusResNet34):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        self.context = ResidualASPP(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))

        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], dim=1))


class HighResolutionUNetPP(UNetPlusPlusResNet34):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        self.high_res_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.high_res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.enc4(x3_0)

        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        x0_3 = x0_3 + self.high_res_scale * self.high_res_branch(x0_0)
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], dim=1))


class GatedAdapterUNetPP(UNetPlusPlusResNet34):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        channels = (64, 64, 128, 256, 512)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            for c in channels
        ])
        self.adapter_gates = nn.Parameter(torch.zeros(len(channels)))

    def _adapt(self, value: torch.Tensor, index: int) -> torch.Tensor:
        return value + self.adapter_gates[index] * self.adapters[index](value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self._adapt(self.stem(x), 0)
        x1_0 = self._adapt(self.enc1(self.pool(x0_0)), 1)
        x2_0 = self._adapt(self.enc2(x1_0), 2)
        x3_0 = self._adapt(self.enc3(x2_0), 3)
        x4_0 = self._adapt(self.enc4(x3_0), 4)

        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], dim=1))


def replace_decoder_bn_with_gn(model: nn.Module, groups: int = 8) -> None:
    decoder_names = ("x3_1", "x2_1", "x1_1", "x0_1", "x2_2", "x1_2", "x0_2", "x1_3", "x0_3", "head")

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                channels = child.num_features
                setattr(module, name, nn.GroupNorm(min(groups, channels), channels))
            else:
                replace(child)

    for name in decoder_names:
        replace(getattr(model, name))


def freeze_encoder_bn(model: nn.Module) -> None:
    prefixes = ("stem", "enc1", "enc2", "enc3", "enc4")
    for name, module in model.named_modules():
        if name.startswith(prefixes) and isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_model(candidate: str, pretrained: bool, checkpoint: str | None) -> nn.Module:
    if candidate == "aspp":
        model = ASPPUNetPP(pretrained=False)
    elif candidate == "highres":
        model = HighResolutionUNetPP(pretrained=False)
    elif candidate == "gated":
        model = GatedAdapterUNetPP(pretrained=False)
    elif candidate in ("bnfreeze", "bngn"):
        model = UNetPlusPlusResNet34(pretrained=False)
        if candidate == "bngn":
            replace_decoder_bn_with_gn(model)
    elif candidate == "convnext":
        model = ConvNeXtTinyUNet(pretrained=pretrained)
    elif candidate == "hrnet":
        model = HRNetW18FPN()
    elif candidate == "effb4":
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights="imagenet" if pretrained else None,
                        in_channels=3, classes=NUM_CLASSES, activation=None)
    elif candidate == "resnet50":
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                        in_channels=3, classes=NUM_CLASSES, activation=None)
    elif candidate == "resnet50pp":
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                                 in_channels=3, classes=NUM_CLASSES, activation=None)
    else:
        raise ValueError(f"unknown candidate: {candidate}")

    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"loaded={checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None,
            loss_mode: str = "ce", focal_gamma: float = 2.0) -> torch.Tensor:
    if class_weights is not None:
        raise ValueError("CE class weights are disabled in this ablation")
    if loss_mode == "focal":
        log_probability = F.log_softmax(logits, dim=1)
        nll = -log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - torch.exp(-nll)).pow(focal_gamma)
        cross_entropy = (focal_factor * nll).mean()
    else:
        cross_entropy = F.cross_entropy(logits, target)
    return cross_entropy + dice_loss(logits, target)


def confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Dict[str, object]:
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def metrics_from_probabilities(probabilities: Sequence[np.ndarray], targets: Sequence[np.ndarray], bias: float) -> Dict[str, object]:
    """Score a cached probability list without repeating GPU inference."""
    predictions = []
    for value in probabilities:
        adjusted = np.log(np.maximum(value, 1e-7))
        adjusted[3] += bias
        predictions.append(adjusted.argmax(axis=0).astype(np.uint8))
    result = metrics(predictions, targets)
    result["al_bias"] = float(bias)
    return result


@torch.no_grad()
def predict_probabilities_whole_image(model: nn.Module, image_path: Path, device,
                                      use_amp: bool, tta_mode: str, scale: float) -> np.ndarray:
    """D4 inference on one scaled image, padded to the encoder stride."""
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    original_size = (image.shape[1], image.shape[0])
    if abs(float(scale) - 1.0) > 1e-6:
        if float(scale) <= 0:
            raise ValueError("scale must be positive")
        scaled_size = (int(round(image.shape[1] * float(scale))),
                       int(round(image.shape[0] * float(scale))))
        image = np.asarray(Image.fromarray(image).resize(
            scaled_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    height, width = image.shape[:2]
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    padded = np.pad(image, ((0, padded_height - height),
                            (0, padded_width - width), (0, 0)), mode="reflect")
    inputs = normalize_image(padded).unsqueeze(0).to(device)
    probability_sum = None
    transforms = views(tta_mode)
    for rotation, flip in transforms:
        view = transform_view(inputs, rotation, flip)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            probabilities = torch.softmax(model(view), dim=1)
        probabilities = inverse_view(probabilities, rotation, flip)
        probability_sum = probabilities.float() if probability_sum is None else probability_sum + probabilities.float()
    probabilities = (probability_sum / float(len(transforms)))[0, :, :height, :width].cpu().numpy()
    restored = np.stack([
        np.asarray(Image.fromarray(probabilities[class_id]).resize(
            original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


@torch.no_grad()
def collect_probabilities(model: nn.Module, samples, args, device, use_amp):
    probabilities, targets = [], []
    model.eval()
    for image_path, mask_path in _local_progress(samples, 'encoder_structure_search'):
        if args.inference_mode == "whole-image":
            value = predict_probabilities_whole_image(
                model, image_path, device, use_amp, args.tta, args.scale)
        else:
            value = predict_probabilities(
                model, image_path, args.patch_size, args.val_stride, device, use_amp,
                args.tile_batch_size, args.tta, args.blend, args.scale,
            )
        probabilities.append(value)
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return probabilities, targets


@torch.no_grad()
def evaluate_model(model: nn.Module, samples, args, device, use_amp, bias: float) -> Dict[str, object]:
    probabilities, targets = collect_probabilities(model, samples, args, device, use_amp)
    return metrics_from_probabilities(probabilities, targets, bias)


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, freeze_bn: bool,
                    class_weights: torch.Tensor | None = None, loss_mode: str = "ce",
                    focal_gamma: float = 2.0, batch_scheduler=None) -> float:
    model.train()
    if freeze_bn:
        freeze_encoder_bn(model)
    total, count = 0.0, 0
    for images, masks in _local_progress(loader, 'encoder_structure_search'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss = loss_fn(model(images), masks, class_weights, loss_mode, focal_gamma)
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
        if batch_scheduler is not None:
            batch_scheduler.step()
        total += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def train(args: argparse.Namespace) -> None:
    if args.eval_every < 1:
        raise ValueError("--eval-every must be at least 1")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    if args.split_manifest:
        if args.checkpoint:
            raise ValueError("Internal holdout requires fresh initialization, not a KoMaP checkpoint")
        manifest = json.loads(Path(args.split_manifest).read_text())
        root = Path(args.data_dir)
        train_samples = [(root / r["image"], root / r["mask"]) for r in manifest["fit"]]
        valid_samples = [(root / r["image"], root / r["mask"]) for r in manifest["internal_valid"]]
        from improve_five_causes import specimen_id
        assert {specimen_id(p) for p, _ in train_samples}.isdisjoint(
            {specimen_id(p) for p, _ in valid_samples})
    dataset = RareAwarePatchDataset(
        train_samples, args.patch_size, args.patches_per_image, True, args.seed,
        scale=args.scale, sampling_mode=args.sampling_mode,
        metallography_augmentation=args.metallography_augmentation)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0),
    )
    model = build_model(args.candidate, args.pretrained, args.checkpoint).to(device)
    freeze_bn = args.candidate in ("bnfreeze", "bngn")
    if freeze_bn:
        freeze_encoder_bn(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.encoder_lr is None and args.decoder_lr is None:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        if args.encoder_lr is None or args.decoder_lr is None:
            raise ValueError("--encoder-lr and --decoder-lr must be provided together")
        encoder_params, decoder_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (encoder_params if name.startswith("encoder.") else decoder_params).append(parameter)
        if not encoder_params or not decoder_params:
            raise ValueError("differential learning rates require non-empty encoder and decoder groups")
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": decoder_params, "lr": args.decoder_lr},
            ],
            weight_decay=args.weight_decay,
        )
    batch_scheduler = args.warmup_updates > 0
    if batch_scheduler:
        total_updates = len(loader) * args.epochs
        warmup_updates = min(args.warmup_updates, total_updates)

        def lr_multiplier(update):
            if update < warmup_updates:
                return max((update + 1) / warmup_updates, 1e-4)
            progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
            return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    class_weights = None
    if args.class_weights is not None:
        raise ValueError("CE class weights are disabled in this ablation")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    best_bias = 0.0
    start = time.time()
    print(
        f"candidate={args.candidate} device={device} gpu={torch.cuda.current_device() if device.type == 'cuda' else 'cpu'} "
        f"patches={len(dataset)} batch={args.batch_size} params={sum(p.numel() for p in model.parameters()):,} "
        f"lr_groups={[(group['lr']) for group in optimizer.param_groups]}",
        flush=True,
    )
    for epoch in _local_progress(range(1, args.epochs + 1), 'encoder_structure_search'):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, scaler, device, use_amp, freeze_bn,
                                     class_weights, args.loss_mode, args.focal_gamma,
                                     scheduler if batch_scheduler else None)
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        # Sparse validation is useful for expensive magnified inputs: it keeps
        # the fixed inference protocol while avoiding a full D4 pass every epoch.
        valid = (evaluate_model(model, valid_samples, args, device, use_amp, args.eval_bias)
                 if should_evaluate else None)
        if not batch_scheduler:
            scheduler.step()
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "valid": valid, "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, out_dir / "last.pt")
        if valid is None:
            print(f"epoch={epoch:02d}/{args.epochs} loss={train_loss:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.3g} "
                  f"valid=skipped time={record['seconds']:.1f}s", flush=True)
        else:
            score = float(valid[args.selection_metric])
            print(
                f"epoch={epoch:02d}/{args.epochs} loss={train_loss:.4f} miou={score:.5f} "
                f"image={valid['image_mean_miou']:.5f} "
                + ",".join(f"{v:.4f}" for v in valid["image_class_iou"].values())
                + f" time={record['seconds']:.1f}s", flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, out_dir / "best.pt")
    # Calibrate Al on the best training checkpoint with a compact, identical
    # sweep for every candidate. Cache probability maps once; re-running D4
    # inference for each threshold makes this tiny sweep needlessly expensive.
    best_payload = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_payload["model"])
    model.eval()
    cached_probabilities, cached_targets = collect_probabilities(model, valid_samples, args, device, use_amp)
    np.savez_compressed(out_dir / "valid_probs.npz", **{
        image_path.stem: probabilities.astype(np.float16)
        for (image_path, _), probabilities in zip(valid_samples, cached_probabilities)
    })
    sweep = []
    for bias in np.arange(-0.80, 0.21, 0.05):
        value = metrics_from_probabilities(cached_probabilities, cached_targets, float(bias))
        sweep.append(value)
    best_calibrated = max(sweep, key=lambda item: item[args.selection_metric])
    summary = {
        "candidate": args.candidate,
        "epochs": args.epochs,
        "best_fixed_miou": best_score,
        "selection_metric": args.selection_metric,
        "split_manifest": args.split_manifest,
        "best_calibrated": best_calibrated,
        "calibration_sweep": sweep,
        "elapsed_seconds": time.time() - start,
        "checkpoint": str(out_dir / "best.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "best_fixed_miou": best_score,
                      "best_calibrated": best_calibrated}, ensure_ascii=False), flush=True)

    # Produce a test submission for every candidate so the final choice can be
    # made without rerunning inference. The calibrated bias is recorded in the
    # candidate summary and is not silently mixed with other candidates.
    if args.split_manifest or args.skip_test_inference:
        return
    test_dir = out_dir / "test_masks"
    test_dir.mkdir(exist_ok=True)
    bias = float(best_calibrated["al_bias"])
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        probabilities = predict_probabilities(
            model, image_path, args.patch_size, args.val_stride, device, use_amp,
            args.tile_batch_size, args.tta, args.blend, args.scale,
        )
        adjusted = np.log(np.maximum(probabilities, 1e-7))
        adjusted[3] += bias
        prediction = adjusted.argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(test_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote test_masks={len(list(test_dir.glob('*.png')))} dir={test_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=ALL_CANDIDATES, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split-manifest")
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_miou_compat"),
                        default="image_mean_miou")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="magnify the image before patch sampling and tiled inference")
    parser.add_argument("--sampling-mode", choices=("class_focused", "uniform_area", "area_weighted"),
                        default="class_focused")
    parser.add_argument("--metallography-augmentation", action="store_true",
                        help="add random affine, gamma/contrast, blur, and noise transforms")
    parser.add_argument("--warmup-updates", type=int, default=0,
                        help="linear LR warm-up steps before cosine decay; 0 keeps epoch cosine")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="validate every N epochs (the final epoch is always evaluated)")
    parser.add_argument("--skip-test-inference", action="store_true",
                        help="do not run test inference after validation/model selection")
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--inference-mode", choices=("tiles", "whole-image"), default="tiles",
                        help="use overlapping tiles or one reflect-padded whole-image forward per D4 view")
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--encoder-lr", type=float, default=None,
                        help="optional encoder LR; must be paired with --decoder-lr")
    parser.add_argument("--decoder-lr", type=float, default=None,
                        help="optional decoder LR; must be paired with --encoder-lr")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-bias", type=float, default=-0.20)
    parser.add_argument("--tta", choices=("legacy", "d4"), default="d4")
    parser.add_argument("--blend", choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--eval-only-checkpoint")
    parser.add_argument("--class-weights", type=float, nargs=4, default=None)
    parser.add_argument("--loss-mode", choices=("ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.split_manifest and args.eval_only_checkpoint:
        raise ValueError("Use the grouped training path for internal validation")
    if args.eval_only_checkpoint:
        # Reuse the normal model builder and output layout for a checkpoint
        # produced by an interrupted training run.
        seed_everything(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda" and not args.no_amp
        model = build_model(args.candidate, args.pretrained, args.checkpoint).to(device)
        payload = torch.load(args.eval_only_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
        samples = collect_split(args.data_dir, "valid")
        probabilities, targets = collect_probabilities(model, samples, args, device, use_amp)
        sweep = [metrics_from_probabilities(probabilities, targets, float(bias))
                 for bias in np.arange(-0.80, 0.21, 0.05)]
        best = max(sweep, key=lambda item: item[args.selection_metric])
        result = {"candidate": args.candidate, "checkpoint": args.eval_only_checkpoint,
                  "best_calibrated": best, "calibration_sweep": sweep}
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "valid_probs.npz", **{
            image_path.stem: value.astype(np.float16)
            for (image_path, _), value in zip(samples, probabilities)
        })
        (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.skip_test_inference:
            print(json.dumps({"candidate": args.candidate, "best_calibrated": best,
                              "test_inference_skipped": True}, ensure_ascii=False), flush=True)
        else:
            test_dir = out_dir / "test_masks"
            test_dir.mkdir(exist_ok=True)
            bias = float(best["al_bias"])
            for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
                value = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device, use_amp,
                                              args.tile_batch_size, args.tta, args.blend, args.scale)
                adjusted = np.log(np.maximum(value, 1e-7))
                adjusted[3] += bias
                encode_mask(adjusted.argmax(axis=0).astype(np.uint8)).save(
                    test_dir / image_path.name.replace("_image.png", "_mask.png"))
            print(json.dumps({"candidate": args.candidate, "best_calibrated": best,
                              "test_masks": len(list(test_dir.glob("*.png")))}, ensure_ascii=False), flush=True)
    else:
        train(args)
