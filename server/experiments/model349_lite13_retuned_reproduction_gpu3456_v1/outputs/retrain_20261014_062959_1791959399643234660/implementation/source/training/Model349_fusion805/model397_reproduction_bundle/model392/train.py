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
from komap_seg.dataset import KomapFullDataset, KomapPatchDataset, modal_image_width, seed_worker
from komap_seg.losses import BoundaryHeadLoss, ComboLoss
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
    p.add_argument(
        "--normalize-width", action="store_true",
        help="Use the training split's modal native width as the per-image pixel-scale target.",
    )
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
    p.add_argument(
        "--ce-class-weights", type=float, nargs=NUM_CLASSES, default=None,
        metavar=("AL", "AL3NI", "PRIMARY_SI", "EUTECTIC_SI"),
        help="Optional weights for CE only, in the fixed class order.",
    )
    p.add_argument(
        "--init-checkpoint", default=None,
        help="Initialize model weights from an inference checkpoint and start a fresh optimizer.",
    )
    p.add_argument("--no-boundary-head", action="store_true", help="Train without the boundary auxiliary head.")
    p.add_argument("--boundary-head-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rng-mode",
        choices=("legacy", "per_item"),
        default="legacy",
        help="legacy preserves historical sampling; per_item seeds crop/augmentation by seed, epoch, and sample index.",
    )
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
    if args.normalize_width and args.val_every != 0:
        raise ValueError(
            "width-normalization experiments require --val-every 0; "
            "the quick validation path does not implement per-image scale normalization"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = Path(args.data_root)
    args.target_width = modal_image_width(root / "train" / "images") if args.normalize_width else 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = KomapPatchDataset(
        root / "train" / "images",
        root / "train" / "masks",
        patch_size=args.patch_size,
        input_scale=args.input_scale,
        target_width=args.target_width,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
        rng_mode=args.rng_mode,
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
    preprocessing_config = {
        "patch_size": args.patch_size,
        "input_scale": args.input_scale,
        "target_width": args.target_width,
        "normalize_width": args.normalize_width,
    }
    encoder_weights = None if args.init_checkpoint else args.encoder_weights
    model = build_model(**model_config, encoder_weights=encoder_weights).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("model_config") != model_config:
            raise ValueError(
                f"initial checkpoint model config {initial.get('model_config')} "
                f"does not match requested config {model_config}"
            )
        model.load_state_dict(initial["state_dict"])
        print(f"initialized from inference checkpoint {args.init_checkpoint}", flush=True)
    ce_class_weights = (
        torch.tensor(args.ce_class_weights, dtype=torch.float32)
        if args.ce_class_weights is not None else None
    )
    seg_loss_fn = ComboLoss(class_weights=ce_class_weights).to(device)
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
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_args = state.get("args")
        if not isinstance(saved_args, dict):
            raise ValueError("resume checkpoint has no saved args; refusing an unverifiable resume")
        saved_args = dict(saved_args)
        current_args = dict(vars(args))
        # Model382 predates this flag; its omitted mode means the historical mode.
        saved_args.setdefault("rng_mode", "legacy")
        current_args.setdefault("rng_mode", "legacy")
        for key in ("output_dir", "resume"):
            saved_args.pop(key, None)
            current_args.pop(key, None)
        mismatches = {
            key: (saved_args.get(key), current_args.get(key))
            for key in set(saved_args) | set(current_args)
            if saved_args.get(key) != current_args.get(key)
        }
        if mismatches:
            raise ValueError(f"resume arguments differ from checkpoint: {mismatches}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        train_ds.rng.setstate(state["dataset_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"resumed from epoch {start_epoch - 1}; next epoch={start_epoch}", flush=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(
        f"{len(train_ds.images)} train images, {len(train_loader)} steps/epoch, device={device}, "
        f"target_width={args.target_width}", flush=True
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (img, mask) in enumerate(__import__("tqdm").tqdm(train_loader, desc=f"epoch {epoch}", mininterval=10)):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_boundary:
                    seg_logits, boundary_logits = forward_with_boundary(model, img)
                    loss = seg_loss_fn(seg_logits, mask) + args.boundary_head_weight * boundary_loss_fn(
                        boundary_logits, mask
                    )
                else:
                    loss = seg_loss_fn(model(img), mask)
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
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(train_loader):
                print(
                    f"epoch {epoch:3d}/{args.epochs} "
                    f"step {batch_index + 1:4d}/{len(train_loader)} "
                    f"loss={epoch_loss / (batch_index + 1):.4f}",
                    flush=True,
                )
        scheduler.step()

        msg = f"epoch {epoch:3d}/{args.epochs} train_loss={epoch_loss / len(train_loader):.4f} time={time.time() - t0:.1f}s"
        if valid_ds is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            ious = quick_validate(model, valid_ds, device)
            msg += f" quick_val_mIoU={ious.mean().item():.4f}"
        print(msg, flush=True)
        save_checkpoint(output_dir / "last.pt", model, model_config, preprocessing_config)
        torch.save({
            "epoch": epoch,
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
