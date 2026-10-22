"""Train the phase-segmentation model (defaults = the released best recipe).

UNet++ + HRNet-W48 (ImageNet), 512 patches cut from 256 native px (input
scale 2), CE + Dice without class weights, plus a training-only
class-agnostic boundary head. The final model is `<output-dir>/last.pt`.

The per-epoch validation printed here is only a quick progress signal: each
valid image goes through the model in one padded pass at native scale,
which doesn't match input_scale=2 training, so its numbers run well below
the real metric. Use evaluate.py (overlap-tile, same as prediction) for
the actual mIoU.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from komap_seg.checkpoint import save_checkpoint
from komap_seg.dataset import KomapFullDataset, KomapPatchDataset, seed_worker
from komap_seg.losses import (
    BoundaryHeadLoss,
    ComboLoss,
    boundary_ce_pixel_weights,
    directional_unlikelihood_loss,
)
from komap_seg.metrics import confusion_matrix, iou_per_class
from komap_seg.model import build_model, forward_with_boundary
from komap_seg.palette import CLASS_NAMES, NUM_CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Folder with train/ and valid/ (see README).")
    p.add_argument("--output-dir", default="runs/best")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--architecture", default="UnetPlusPlus")
    p.add_argument("--encoder-name", default="tu-hrnet_w48")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--input-scale", type=float, default=2.0, help="Must match at evaluate/predict time.")
    p.add_argument("--samples-per-image", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--accumulation-steps", type=int, default=1,
        help="Micro-batches per optimizer update (use 4 with batch-size 2 to keep effective batch 8).",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr-mult", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-boundary-head", action="store_true", help="Train without the boundary auxiliary head.")
    p.add_argument("--boundary-head-weight", type=float, default=1.0)
    p.add_argument(
        "--boundary-ce-weight", type=float, default=0.0,
        help="Maximum Gaussian CE boost on GT boundaries; 0 preserves the released loss exactly.",
    )
    p.add_argument("--boundary-ce-radius", type=int, default=3)
    p.add_argument("--boundary-ce-sigma", type=float, default=1.5)
    p.add_argument(
        "--boundary-ce-warmup-fraction", type=float, default=0.1,
        help="Fraction of all training steps to linearly warm the boundary CE boost.",
    )
    p.add_argument(
        "--directional-unlikelihood-weight", type=float, default=0.0,
        help="Max coefficient for Eutectic-side Al3Ni suppression at their GT interface.",
    )
    p.add_argument(
        "--directional-unlikelihood-warmup-fraction", type=float, default=0.1,
        help="Fraction of all training steps to linearly warm the directional loss.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-every", type=int, default=5, help="0 = no per-epoch validation.")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA FP16 autocast.")
    p.add_argument("--resume", action="store_true", help="Resume from output-dir/train_state.pt when present.")
    return p.parse_args()


@torch.no_grad()
def quick_validate(model, valid_ds: KomapFullDataset, device: torch.device) -> torch.Tensor:
    """Per-class IoU from one padded forward per valid image (progress signal only)."""
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for sample in valid_ds:
        image = sample["image"].unsqueeze(0).to(device)
        h, w = image.shape[-2:]
        pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32
        padded = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        logits = model(padded)[..., :h, :w]
        conf += confusion_matrix(logits.argmax(1).cpu(), sample["mask"], NUM_CLASSES)
    model.train()
    return iou_per_class(conf)


def main() -> None:
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be >= 1")
    if args.boundary_ce_weight < 0 or args.boundary_ce_radius < 0 or args.boundary_ce_sigma <= 0:
        raise ValueError("boundary CE weight/radius must be nonnegative and sigma positive")
    if not 0 <= args.boundary_ce_warmup_fraction <= 1:
        raise ValueError("--boundary-ce-warmup-fraction must be in [0, 1]")
    if args.directional_unlikelihood_weight < 0:
        raise ValueError("--directional-unlikelihood-weight must be nonnegative")
    if not 0 <= args.directional_unlikelihood_warmup_fraction <= 1:
        raise ValueError("--directional-unlikelihood-warmup-fraction must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_ds = KomapPatchDataset(
        root / "train" / "images",
        root / "train" / "masks",
        patch_size=args.patch_size,
        input_scale=args.input_scale,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    valid_ds = KomapFullDataset(root / "valid" / "images", root / "valid" / "masks") if args.val_every > 0 else None

    use_boundary = not args.no_boundary_head
    model_config = {
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
        "boundary_head": use_boundary,
    }
    model = build_model(**model_config, encoder_weights=args.encoder_weights).to(device)
    seg_loss_fn = ComboLoss().to(device)
    boundary_loss_fn = BoundaryHeadLoss().to(device)

    encoder_ids = {id(p) for p in model.encoder.parameters()}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr * args.encoder_lr_mult},
            {"params": decoder_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    state_path = output_dir / "train_state.pt"
    start_epoch = 1
    global_step = 0
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", (start_epoch - 1) * len(train_loader)))
        train_ds.rng.setstate(state["dataset_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"resumed from epoch {start_epoch - 1}; next epoch={start_epoch}", flush=True)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.boundary_ce_warmup_fraction)
    directional_warmup_steps = int(total_steps * args.directional_unlikelihood_warmup_fraction)
    print(
        f"{len(train_ds.images)} train images, {len(train_loader)} steps/epoch, device={device}, "
        f"boundary_ce_max={args.boundary_ce_weight:g}, warmup_steps={warmup_steps}, "
        f"directional_unlikelihood_max={args.directional_unlikelihood_weight:g}, "
        f"directional_warmup_steps={directional_warmup_steps}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        last_boundary_ce_weight = 0.0
        directional_raw_sum = torch.zeros((), device=device, dtype=torch.float32)
        directional_weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        directional_support_sum = torch.zeros((), device=device, dtype=torch.long)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            if args.boundary_ce_weight > 0:
                if warmup_steps > 0:
                    ramp = min((global_step + 1) / warmup_steps, 1.0)
                else:
                    ramp = 1.0
                current_boundary_ce_weight = args.boundary_ce_weight * ramp
                ce_pixel_weights = boundary_ce_pixel_weights(
                    mask,
                    weight=current_boundary_ce_weight,
                    radius=args.boundary_ce_radius,
                    sigma=args.boundary_ce_sigma,
                )
            else:
                current_boundary_ce_weight = 0.0
                ce_pixel_weights = None
            last_boundary_ce_weight = current_boundary_ce_weight
            if args.directional_unlikelihood_weight > 0:
                if directional_warmup_steps > 0:
                    directional_ramp = min((global_step + 1) / directional_warmup_steps, 1.0)
                else:
                    directional_ramp = 1.0
                current_directional_weight = args.directional_unlikelihood_weight * directional_ramp
            else:
                current_directional_weight = 0.0
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_boundary:
                    seg_logits, boundary_logits = forward_with_boundary(model, img)
                    loss = seg_loss_fn(seg_logits, mask, ce_pixel_weights=ce_pixel_weights)
                    loss = loss + args.boundary_head_weight * boundary_loss_fn(boundary_logits, mask)
                else:
                    seg_logits = model(img)
                    loss = seg_loss_fn(seg_logits, mask, ce_pixel_weights=ce_pixel_weights)
                if current_directional_weight > 0:
                    directional_raw, directional_support = directional_unlikelihood_loss(seg_logits, mask)
                    directional_term = current_directional_weight * directional_raw
                    loss = loss + directional_term
                else:
                    directional_raw = loss.detach().new_zeros(())
                    directional_term = directional_raw
                    directional_support = torch.zeros((), device=device, dtype=torch.long)
            directional_raw_sum += directional_raw.detach()
            directional_weighted_sum += directional_term.detach()
            directional_support_sum += directional_support.detach()
            scaler.scale(loss / args.accumulation_steps).backward()
            is_update = (
                (batch_index + 1) % args.accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            global_step += 1
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(train_loader):
                print(
                    f"epoch {epoch:3d}/{args.epochs} "
                    f"step {batch_index + 1:4d}/{len(train_loader)} "
                    f"loss={epoch_loss / (batch_index + 1):.4f} "
                    f"boundary_ce={current_boundary_ce_weight:.3f} "
                    f"dir_beta={current_directional_weight:.3f} "
                    f"dir_raw_mean={(directional_raw_sum / (batch_index + 1)).item():.6f} "
                    f"dir_term_mean={(directional_weighted_sum / (batch_index + 1)).item():.6f} "
                    f"dir_support_px_per_patch="
                    f"{(directional_support_sum.float() / ((batch_index + 1) * img.shape[0])).item():.1f}",
                    flush=True,
                )
        scheduler.step()

        msg = (
            f"epoch {epoch:3d}/{args.epochs} train_loss={epoch_loss / len(train_loader):.4f} "
            f"boundary_ce={last_boundary_ce_weight:.3f} "
            f"dir_beta={current_directional_weight:.3f} "
            f"dir_raw_mean={(directional_raw_sum / len(train_loader)).item():.6f} "
            f"dir_term_mean={(directional_weighted_sum / len(train_loader)).item():.6f} "
            f"dir_support_px_per_patch="
            f"{(directional_support_sum.float() / (len(train_loader) * args.batch_size)).item():.1f} "
            f"dir_loss_share_pct="
            f"{(100.0 * directional_weighted_sum / max(epoch_loss, 1e-12)).item():.3f} "
            f"time={time.time() - t0:.1f}s"
        )
        if valid_ds is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            ious = quick_validate(model, valid_ds, device)
            msg += f" quick_val_mIoU={ious.mean().item():.4f}"
        print(msg, flush=True)
        save_checkpoint(output_dir / "last.pt", model, model_config)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "dataset_rng_state": train_ds.rng.getstate(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "args": vars(args),
        }, state_path)

    print(f"done. final model: {output_dir / 'last.pt'} (evaluate it with evaluate.py)")


if __name__ == "__main__":
    main()
