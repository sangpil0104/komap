"""Train-only brightness diagnostics, illustrative photometric examples. No training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from alphase.io import ROOT, jsonl_read, json_write
from alphase.contracts import CLASS_NAMES, rgb_to_ids
from alphase.data import native_photo

rows = [
    r for r in jsonl_read(ROOT / "manifests/dataset.jsonl") if r["split"] == "train"
]
original = [[] for _ in range(4)]
augmented = [[] for _ in range(4)]
out = ROOT / "reports/train_augmentation"
out.mkdir(exist_ok=True)
for index, row in enumerate(tqdm(rows, desc="train-only photo audit")):
    gray = np.array(Image.open(row["image_path"]).convert("L"))
    target = rgb_to_ids(np.array(Image.open(row["mask_path"])))
    changed = native_photo(gray, 2026 + index, False)
    for c in range(4):
        if (target == c).any():
            original[c].append(float(gray[target == c].mean()))
            augmented[c].append(float(changed[target == c].mean()))
    if index < 4:
        h, w = gray.shape
        canvas = Image.new("L", (2 * w, h + 40), 255)
        canvas.paste(Image.fromarray(gray), (0, 40))
        canvas.paste(Image.fromarray(changed), (w, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), row["image_id"] + " / original", fill=0)
        draw.text(
            (w + 8, 8),
            "Native photometric example (seed " + str(2026 + index) + ")",
            fill=0,
        )
        canvas.save(out / (row["image_id"] + ".png"))
result = {
    "source_split": "train",
    "image_count": 70,
    "augmentation_seed": 2026,
    "blur_noise": False,
    "role": "diagnostic only; class GT never used for input normalization",
    "classes": {
        name: {
            "original_image_mean_min_median_max": np.quantile(
                original[c], [0, 0.5, 1]
            ).tolist(),
            "augmented_image_mean_min_median_max": np.quantile(
                augmented[c], [0, 0.5, 1]
            ).tolist(),
        }
        for c, name in enumerate(CLASS_NAMES)
    },
}
json_write(out / "statistics.json", result)
