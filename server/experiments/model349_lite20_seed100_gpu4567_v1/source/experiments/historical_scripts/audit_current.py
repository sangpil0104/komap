#!/usr/bin/env python3
"""Print per-image/class errors for the current 768 ensemble."""
from pathlib import Path
import numpy as np
from PIL import Image
from aspp768_addition import BIASES, MATRIX, cleanup_primary
from improve_five_causes import collect_split, decode_mask

cache = Path("outputs/restart52_aspp768_addition/cache")
names = ["Al3Ni", "Primary", "Al", "Eutectic"]
rows = []
for image_path, mask_path in collect_split("data", "valid"):
    loaded = np.load(cache / f"{image_path.stem}.npz")
    values = [loaded[f"p{i}"].astype(np.float32) for i in range(4)]
    prior = loaded["prior"].astype(np.float32)
    weight = .6
    mixed = np.asarray([sum(MATRIX[c,m]*(1-weight)*values[m][c] for m in range(3))+weight*values[3][c]
                        for c in range(4)], np.float32)
    mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
    logits = np.log(np.maximum(mixed, 1e-7)) + .09*prior + BIASES[:,None,None]
    pred = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    cm = np.bincount(target.reshape(-1)*4+pred.reshape(-1), minlength=16).reshape(4,4)
    inter=np.diag(cm).astype(float); union=cm.sum(0)+cm.sum(1)-inter
    iou=np.divide(inter, union, out=np.zeros(4), where=union>0)
    rows.append((float(iou.mean()), image_path.name, iou, target, pred, cm))
for score,name,iou,target,pred,cm in sorted(rows):
    target_counts=np.bincount(target.reshape(-1),minlength=4)
    pred_counts=np.bincount(pred.reshape(-1),minlength=4)
    print(f"{score:.6f} {name} iou=" + ",".join(f"{x:.4f}" for x in iou)
          + " target=" + ",".join(map(str,target_counts.tolist()))
          + " pred=" + ",".join(map(str,pred_counts.tolist())))
    print("  cm=" + np.array2string(cm, separator=",", max_line_width=300))
