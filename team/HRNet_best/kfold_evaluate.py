"""Out-of-fold image mIoU over all labeled images.

Each fold's model is evaluated only on that fold's held-out images, so every
labeled image is scored exactly once by a model that never trained on it.
Uses the same overlap-tile + flip TTA path as evaluate.py / predict.py.
"""

import argparse
from pathlib import Path

import torch

from komap_seg.checkpoint import load_model
from komap_seg.dataset import KomapFullDataset
from komap_seg.inference import predict_probs_tta
from komap_seg.metrics import confusion_matrix, image_iou, iou_per_class
from komap_seg.palette import CLASS_NAMES, NUM_CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds-root", default="runs/folds", help="make_folds.py output (fold{i}/valid/...).")
    p.add_argument("--runs-root", default="runs/kfold", help="Where fold{i}/<checkpoint-name> live.")
    p.add_argument("--checkpoint-name", default="last.pt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=170)
    p.add_argument("--input-scale", type=float, default=2.0, help="Must match training.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    all_images = []
    for i in range(args.k):
        model = load_model(Path(args.runs_root) / f"fold{i}" / args.checkpoint_name, device)
        fold = Path(args.folds_root) / f"fold{i}" / "valid"
        ds = KomapFullDataset(fold / "images", fold / "masks")
        fold_images = []
        for sample in ds:
            probs = predict_probs_tta(
                model,
                sample["image"],
                patch_size=args.patch_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                tta=not args.no_tta,
                input_scale=args.input_scale,
                device=device,
            )
            conf = confusion_matrix(probs.argmax(0), sample["mask"], NUM_CLASSES)
            total_conf += conf
            fold_images.append(image_iou(conf))
        all_images.extend(fold_images)
        fi = torch.stack(fold_images)
        print(f"fold{i} ({len(ds)} held-out): image mIoU={fi.nanmean(1).mean().item():.4f}  "
              + " ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, fi.nanmean(0).tolist())), flush=True)

    img = torch.stack(all_images)
    print(f"\nout-of-fold, {len(all_images)} images, per-class IoU (mean over images):")
    for name, iou in zip(CLASS_NAMES, img.nanmean(0).tolist()):
        print(f"  {name:12s} {iou:.4f}")
    print(f"cross-validated image mIoU = {img.nanmean(1).mean().item():.4f}")
    print(f"(reference) cross-validated global mIoU = {iou_per_class(total_conf).mean().item():.4f}")


if __name__ == "__main__":
    main()
