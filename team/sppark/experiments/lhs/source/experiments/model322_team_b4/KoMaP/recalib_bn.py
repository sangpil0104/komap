"""Recompute BatchNorm running statistics after training with a tiny batch size.

Training with bs=1 (plus gradient accumulation) leaves BN running stats estimated from
single crops, which makes inference unstable. This re-estimates them with forward passes
only -- weights are untouched -- and re-evaluates on valid.
"""
import argparse
import json
import os

import albumentations as A
import cv2
import numpy as np
import torch

from common import (build_model, image_miou, list_split, load_gray, load_mask, make_input,
                    predict_probs, resize_gray)

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--data", default=os.path.expanduser("~/komap/data"))
p.add_argument("--weights", default="last.pt")
p.add_argument("--out_weights", default="bnrecal.pt")
p.add_argument("--batches", type=int, default=100)
p.add_argument("--bs", type=int, default=2)
p.add_argument("--momentum", type=float, default=0.02)
args = p.parse_args()

a = json.load(open(os.path.join(args.run, "args.json")))
model = build_model(a["arch"], a["encoder"], pretrained=False)
model.load_state_dict(torch.load(os.path.join(args.run, args.weights), map_location="cpu"))
model = model.cuda()

crop, scale, norm = a["crop"], a.get("scale", 1.0), a["norm"]
train_pairs = [pm for s in a["train_splits"].split(",") for pm in list_split(args.data, s)]
imgs = [load_gray(i) for i, _ in train_pairs]
area = np.array([im.size for im in imgs], np.float64)
w = area / area.sum()
aug = A.Compose([
    A.Affine(scale=(0.7, 1.4), rotate=(-180, 180), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomCrop(crop, crop),
    A.D4(),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
    A.RandomGamma(gamma_limit=(60, 160), p=0.5),
])


def sample_batch(n):
    xs = []
    for _ in range(n):
        im = imgs[np.random.choice(len(imgs), p=w)]
        s = int(crop * 2.1 / scale)
        H, W = im.shape
        if H > s and W > s:
            y, x = np.random.randint(0, H - s + 1), np.random.randint(0, W - s + 1)
            im = im[y:y + s, x:x + s]
        im = resize_gray(im, scale) if scale != 1 else im
        xs.append(make_input(aug(image=im)["image"], norm, a.get("input", "gray3"), a.get("feat_stats")))
    return torch.from_numpy(np.stack(xs)).cuda()


bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
print(f"{len(bns)} BatchNorm layers; {args.batches} batches of {args.bs} crops at {crop}px", flush=True)
for m in bns:
    m.reset_running_stats()
    m.momentum = args.momentum
model.train()
with torch.no_grad():
    for i in range(args.batches):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(sample_batch(args.bs))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.batches}", flush=True)
model.eval()
torch.save(model.state_dict(), os.path.join(args.run, args.out_weights))

rows = []
for ip, mp in list_split(args.data, "valid"):
    pred = predict_probs([(model, norm, scale, a.get("input", "gray3"), a.get("feat_stats"))],
                         load_gray(ip)).argmax(0).cpu().numpy()
    rows.append(image_miou(pred, load_mask(mp)))
per_class = np.nanmean(np.stack([r[2] for r in rows]), 0) * 100
res = {"miou_tta": float(np.mean([r[0] for r in rows]) * 100),
       "miou_tta_absent1": float(np.mean([r[1] for r in rows]) * 100),
       "per_class_iou": per_class.round(2).tolist()}
json.dump(res, open(os.path.join(args.run, "valid_bnrecal.json"), "w"), indent=2)
print(f"[valid TTA after BN recalibration] mIoU {res['miou_tta']:.2f} "
      f"(absent=1: {res['miou_tta_absent1']:.2f}) per-class {res['per_class_iou']}", flush=True)
