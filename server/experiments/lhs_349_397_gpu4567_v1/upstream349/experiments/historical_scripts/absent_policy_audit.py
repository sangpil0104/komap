#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from PIL import Image

from aspp768_addition import BIASES, MATRIX, cleanup_primary
from improve_five_causes import collect_split, decode_mask


cache = Path("outputs/restart52_aspp768_addition/cache")
zero_scores, one_scores = [], []
for image_path, mask_path in collect_split("data", "valid"):
    loaded = np.load(cache / f"{image_path.stem}.npz")
    values = [loaded[f"p{i}"].astype(np.float32) for i in range(4)]
    prior = loaded["prior"].astype(np.float32)
    weight = 0.6
    mixed = np.asarray([
        sum(MATRIX[c, m] * (1.0 - weight) * values[m][c] for m in range(3))
        + weight * values[3][c] for c in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior + BIASES[:, None, None]
    prediction = cleanup_primary(logits.argmax(axis=0).astype(np.uint8), logits, 128)
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    confusion = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(float)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    values = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
    zero_scores.append(float(values.mean()))
    one_scores.append(float(np.where(union == 0, 1.0, values).mean()))
    absent = np.where(union == 0)[0].tolist()
    if absent:
        print(image_path.name, "absent_union_classes", absent,
              "zero", float(values.mean()), "one", float(np.where(union == 0, 1.0, values).mean()))
print("zero_policy", float(np.mean(zero_scores)))
print("absent_as_one", float(np.mean(one_scores)))
print("delta", float(np.mean(one_scores) - np.mean(zero_scores)))
