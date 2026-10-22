"""Side-by-side panel: input / GT green / Otsu-baseline green / model green, for a few valid images."""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from baseline_otsu import postprocess, threshold_masks
from common import build_model, list_split, load_gray, load_mask, make_member, predict_probs

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.expanduser("~/komap/data"))
p.add_argument("--run", default=os.path.expanduser("~/komap/runs/r1_unet_effb4_imnet"))
p.add_argument("--otsu", default=os.path.expanduser("~/komap/runs/otsu/green_baseline.json"))
p.add_argument("--names", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--width", type=int, default=520)
args = p.parse_args()

sel = json.load(open(args.otsu))["selected"]
a = json.load(open(os.path.join(args.run, "args.json")))
model = build_model(a["arch"], a["encoder"], pretrained=False)
model.load_state_dict(torch.load(os.path.join(args.run, "last.pt"), map_location="cpu"))
model = model.cuda().eval()

pairs = {os.path.basename(i).replace("_image.png", ""): (i, m) for i, m in list_split(args.data, "valid")}
rows = []
for name in args.names:
    ip, mp = pairs[name]
    gray = load_gray(ip)
    gt = load_mask(mp) == 2
    otsu = postprocess(threshold_masks(gray)[sel["method"]], sel["open"], sel["min_size"])
    member = make_member(model, a["norm"], a.get("scale", 1.0), a.get("input", "gray3"),
                         a.get("feat_stats"), a.get("target_width", 0))
    pred = predict_probs([member], gray).argmax(0).cpu().numpy() == 2
    panels = [np.stack([gray] * 3, -1)] + [np.where(m[..., None], np.array([76, 178, 76], np.uint8),
                                                    np.stack([gray // 3 + 170] * 3, -1)) for m in (gt, otsu, pred)]
    rows.append(np.concatenate([p.astype(np.uint8) for p in panels], 1))

W = args.width * 4
imgs = [Image.fromarray(r).resize((W, int(r.shape[0] * W / r.shape[1])), Image.LANCZOS) for r in rows]
canvas = Image.new("RGB", (W, sum(i.height for i in imgs)), "white")
y = 0
for i in imgs:
    canvas.paste(i, (0, y)); y += i.height
canvas.save(args.out)
print("wrote", args.out, "(columns: input | GT | Otsu | model)")
