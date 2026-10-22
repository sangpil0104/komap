"""Classical image-processing baseline for the green phase (class 2, dark polygonal particles).

Green particles are dark AND compact, while class-3 needles are dark but thin, so pure intensity
thresholding needs a shape/size filter to separate them. Threshold maps are computed once per
(image, method); the post-processing grid is tuned on train and the selected setting is reported
on valid. An oracle threshold (best possible per image) shows the ceiling of intensity-only
thresholding.
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu

from common import list_split, load_gray, load_mask

GREEN = 2
METHODS = ["otsu", "multi3", "multi4"]  # 5-class multi-Otsu is too slow to be a sensible baseline


def binary_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return np.nan if union == 0 else inter / union


def threshold_masks(gray):
    """Boolean 'dark enough to maybe be green' map for each method."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    out = {"otsu": g <= threshold_otsu(g)}
    for k in (3, 4):
        out[f"multi{k}"] = g <= threshold_multiotsu(g, classes=k, nbins=128)[0]
    return out


def postprocess(mask, open_r, min_size):
    m = mask.astype(np.uint8)
    if open_r:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    if min_size:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_size
        m = keep[lab]
    return m.astype(bool)


def prepare(pairs):
    data = []
    for ip, mp in pairs:
        gray = load_gray(ip)
        data.append((os.path.basename(ip).replace("_image.png", ""), gray,
                     load_mask(mp) == GREEN, threshold_masks(gray)))
        print(".", end="", flush=True)
    print()
    return data


def score(data, method, open_r, min_size):
    ious = {n: binary_iou(postprocess(th[method], open_r, min_size), gt) for n, _, gt, th in data}
    return np.nanmean(np.array(list(ious.values()), float)), ious


def oracle(data):
    """Best achievable IoU with a per-image global grayscale threshold (cheating upper bound)."""
    out = {}
    for n, gray, gt, _ in data:
        g = cv2.GaussianBlur(gray, (0, 0), 1.0)
        best = 0.0
        for t in range(0, 256, 2):
            iou = binary_iou(g <= t, gt)
            best = max(best, 0.0 if np.isnan(iou) else iou)
        out[n] = best if gt.any() else np.nan
    return np.nanmean(np.array(list(out.values()), float)), out



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/komap/data"))
    p.add_argument("--out", default=os.path.expanduser("~/komap/runs/otsu"))
    args = p.parse_args()
    print("preparing train", flush=True)
    train = prepare(list_split(args.data, "train"))
    print("preparing valid", flush=True)
    valid = prepare(list_split(args.data, "valid"))

    results = []
    for method, open_r, min_size in itertools.product(METHODS, [0, 1, 2, 3], [0, 50, 200, 800, 3000]):
        s, _ = score(train, method, open_r, min_size)
        results.append((s, method, open_r, min_size))
        print(f"train green IoU {s*100:6.2f}  {method} open{open_r} min{min_size}", flush=True)

    results.sort(reverse=True)
    best = results[0]
    print(f"\nselected on train: {best[1]} open{best[2]} min{best[3]} -> train {best[0]*100:.2f}")

    vm, vi = score(valid, best[1], best[2], best[3])
    om, oi = oracle(valid)
    otm, _ = oracle(train)
    print(f"VALID green IoU (Otsu baseline): {vm*100:.2f}")
    print(f"VALID green IoU (oracle per-image threshold): {om*100:.2f}   [train {otm*100:.2f}]")
    for k, v in sorted(vi.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1])):
        print(f"  {k:<26} otsu {v*100:6.2f}   oracle {oi[k]*100:6.2f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump({"grid_train": [{"iou": r[0] * 100, "method": r[1], "open": r[2], "min_size": r[3]} for r in results],
               "selected": {"method": best[1], "open": best[2], "min_size": best[3], "train_iou": best[0] * 100},
               "valid_mean": vm * 100, "valid_per_image": {k: None if np.isnan(v) else v * 100 for k, v in vi.items()},
               "valid_oracle_mean": om * 100,
               "valid_oracle_per_image": {k: None if np.isnan(v) else v * 100 for k, v in oi.items()}},
              open(os.path.join(args.out, "green_baseline.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
