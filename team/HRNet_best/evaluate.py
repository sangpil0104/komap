"""Full-resolution mIoU on a labeled split, with the same overlap-tile + flip
TTA path that predict.py uses.

Headline metric: **image mIoU** -- for each image, the mean IoU over the
classes present in its ground truth or prediction, then averaged over
images. Global mIoU (confusion matrix summed over all images first) is
printed for reference only.
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
    p.add_argument("--checkpoint", nargs="+", required=True, help="One or more .pt files; several are ensembled.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="valid", choices=["train", "valid"])
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
    root = Path(args.data_root)
    models = [load_model(Path(c), device) for c in args.checkpoint]
    ds = KomapFullDataset(root / args.split / "images", root / args.split / "masks")

    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    per_image = []
    for sample in ds:
        probs = sum(
            predict_probs_tta(
                m,
                sample["image"],
                patch_size=args.patch_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                tta=not args.no_tta,
                input_scale=args.input_scale,
                device=device,
            )
            for m in models
        )
        conf = confusion_matrix(probs.argmax(0), sample["mask"], NUM_CLASSES)
        total_conf += conf
        per_image.append(image_iou(conf))
        print(f"{sample['stem']:24s} {sample['width']}x{sample['height']}  image mIoU={per_image[-1].nanmean().item():.4f}", flush=True)

    img = torch.stack(per_image)
    print("\nper-class IoU (mean over images):")
    for name, iou in zip(CLASS_NAMES, img.nanmean(0).tolist()):
        print(f"  {name:12s} {iou:.4f}")
    print(f"image mIoU = {img.nanmean(1).mean().item():.4f}")
    g = iou_per_class(total_conf)
    print(f"(reference) global mIoU = {g.mean().item():.4f}")


if __name__ == "__main__":
    main()
