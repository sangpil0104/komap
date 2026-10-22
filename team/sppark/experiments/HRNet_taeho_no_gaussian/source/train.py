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
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from komap_seg.checkpoint import save_checkpoint
from komap_seg.dataset import KomapFullDataset, KomapPatchDataset, seed_worker
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
    p.add_argument("--samples-per-image", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr-mult", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-boundary-head", action="store_true", help="Train without the boundary auxiliary head.")
    p.add_argument("--boundary-head-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-every", type=int, default=5, help="0 = no per-epoch validation.")
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
    print(f"{len(train_ds.images)} train images, {len(train_loader)} steps/epoch, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for img, mask in train_loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_boundary:
                seg_logits, boundary_logits = forward_with_boundary(model, img)
                loss = seg_loss_fn(seg_logits, mask) + args.boundary_head_weight * boundary_loss_fn(
                    boundary_logits, mask
                )
            else:
                loss = seg_loss_fn(model(img), mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        msg = f"epoch {epoch:3d}/{args.epochs} train_loss={epoch_loss / len(train_loader):.4f} time={time.time() - t0:.1f}s"
        if valid_ds is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            ious = quick_validate(model, valid_ds, device)
            msg += f" quick_val_mIoU={ious.mean().item():.4f}"
        print(msg, flush=True)
        save_checkpoint(output_dir / "last.pt", model, model_config)

    print(f"done. final model: {output_dir / 'last.pt'} (evaluate it with evaluate.py)")


if __name__ == "__main__":
    main()
