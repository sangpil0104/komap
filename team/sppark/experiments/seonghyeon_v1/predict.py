"""Ensemble inference: writes color masks and (if GT exists) per-image mIoU."""
import argparse
import json
import os

import numpy as np
from PIL import Image

from common import image_miou, list_split, load_gray, load_mask, load_member, mask_to_rgb, predict_probs

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.expanduser("~/komap/data"))
p.add_argument("--split", default="test")
p.add_argument("--runs", nargs="+", required=True, help="run dirs containing args.json and last.pt")
p.add_argument("--out", required=True)
p.add_argument("--no_tta", action="store_true")
p.add_argument("--device", default="cuda")
p.add_argument("--weights", default="last.pt", help="last.pt or ema.pt")
p.add_argument("--scales", default="1", help="multi-scale TTA, e.g. 0.9,1,1.1 (multiplies each run's scale)")
args = p.parse_args()

members = []
for r in args.runs:
    mem = load_member(r, args.device, args.weights)
    members += [dict(mem, scale_mult=float(f)) for f in args.scales.split(",")]

os.makedirs(args.out, exist_ok=True)
scores = {}
for img_path, mask_path in list_split(args.data, args.split):
    name = os.path.basename(img_path).replace("_image.png", "")
    pred = predict_probs(members, load_gray(img_path), tta=not args.no_tta,
                         device=args.device).argmax(0).cpu().numpy().astype(np.uint8)
    Image.fromarray(mask_to_rgb(pred)).save(os.path.join(args.out, name + "_mask.png"))
    print(f"{name}: saved", flush=True)
    if mask_path:
        a, b, iou = image_miou(pred, load_mask(mask_path))
        scores[name] = {"miou": a * 100, "miou_absent1": b * 100, "iou": (iou * 100).round(2).tolist()}
        print(f"{name}: mIoU {a*100:.2f}  per-class {np.round(iou*100, 1)}")
if scores:
    mean = np.mean([s["miou"] for s in scores.values()])
    mean1 = np.mean([s["miou_absent1"] for s in scores.values()])
    print(f"MEAN mIoU over {len(scores)} images: {mean:.2f} (absent=1: {mean1:.2f})")
    json.dump({"mean_miou": mean, "mean_miou_absent1": mean1, "images": scores},
              open(os.path.join(args.out, "scores.json"), "w"), indent=2)
