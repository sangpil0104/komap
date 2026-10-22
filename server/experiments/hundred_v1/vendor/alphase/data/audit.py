from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from alphase.data.palette import PALETTE, decode_mask
from alphase.utils import json_hash, sha256, write_json


def audit_dataset(root, out_dir, strict_counts=True):
    root, out_dir = Path(root).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sizes, counts = [], defaultdict(lambda: defaultdict(int)), {}
    file_hashes, pixel_hashes = defaultdict(list), defaultdict(list)
    class_counts = np.zeros(4, dtype=np.int64)
    seen = set()
    for split, expected in (("train", 70), ("valid", 20), ("test", 10)):
        paths = sorted((root / split / "images").glob("*_image.png"))
        if strict_counts and len(paths) != expected:
            raise ValueError(f"{split}: expected {expected} images, found {len(paths)}")
        masks = {p.name.removesuffix("_mask.png"): p for p in (root / split / "masks").glob("*_mask.png")}
        ids = {p.name.removesuffix("_image.png") for p in paths}
        if (split != "test" and ids != set(masks)) or (split == "test" and masks):
            raise ValueError(f"Unmatched images/masks in {split}")
        expected_files = set(paths) | set(masks.values())
        unexpected = [str(p) for p in (root / split).rglob("*") if p.is_file() and p not in expected_files]
        if unexpected:
            raise ValueError(f"Unexpected files: {unexpected}")
        counts[split] = len(paths)
        for path in tqdm(paths, desc=f"Audit {split}"):
            image_id = path.name.removesuffix("_image.png")
            if image_id in seen:
                raise ValueError(f"Duplicate image ID: {image_id}")
            seen.add(image_id)
            with Image.open(path) as im:
                mode = im.mode
                arr = np.asarray(im.convert("RGBA"))
            if mode != "RGBA" or not np.all(arr[..., 3] == 255):
                raise ValueError(f"Expected opaque RGBA input: {path}")
            if not np.array_equal(arr[..., 0], arr[..., 1]) or not np.array_equal(arr[..., 1], arr[..., 2]):
                raise ValueError(f"Expected grayscale RGB input: {path}")
            height, width = arr.shape[:2]
            pixel_hash = hashlib.sha256(arr.tobytes()).hexdigest()
            file_hash = sha256(path)
            file_hashes[file_hash].append(image_id)
            pixel_hashes[pixel_hash].append(image_id)
            match = re.fullmatch(r"(A\d+)_(step\d+)_(sample\d+)", image_id)
            if match is None:
                raise ValueError(f"Unexpected image ID: {image_id}")
            row = dict(image_id=image_id, source_split=split, image_path=str(path),
                       mask_path="", width=width, height=height, group_id=match[1],
                       step_token=match[2], sample_token=match[3], image_sha256=file_hash,
                       image_pixel_sha256=pixel_hash, mask_sha256="", image_mode=mode,
                       gray_mean=float(arr[..., 0].mean()), gray_std=float(arr[..., 0].std()))
            for c in range(4):
                row[f"phase{c}_fraction"] = None
            if split != "test":
                target = decode_mask(masks[image_id])
                if target.shape != (height, width):
                    raise ValueError(f"Image/mask size mismatch: {image_id}")
                with Image.open(masks[image_id]) as im:
                    if im.mode != "RGBA" or not np.array_equal(np.asarray(im.convert("RGB")), PALETTE[target]):
                        raise ValueError(f"Mask roundtrip failure: {image_id}")
                n = np.bincount(target.ravel(), minlength=4)
                class_counts += n
                row.update(mask_path=str(masks[image_id]), mask_sha256=sha256(masks[image_id]))
                row.update({f"phase{c}_fraction": float(n[c] / target.size) for c in range(4)})
                cache = out_dir / "masks"
                cache.mkdir(exist_ok=True)
                np.save(cache / f"{image_id}.npy", target.astype(np.uint8))
            rows.append(row)
            sizes[f"{width}x{height}"][split] += 1
    duplicates = {"files": [v for v in file_hashes.values() if len(v) > 1],
                  "pixels": [v for v in pixel_hashes.values() if len(v) > 1]}
    if duplicates["pixels"]:
        raise ValueError(f"Duplicate input pixels require split review: {duplicates}")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "manifest.csv", index=False)
    dataset_hash = json_hash([(r["image_id"], r["source_split"], r["image_sha256"], r["mask_sha256"]) for r in rows])
    groups = {s: set(table.loc[table.source_split == s, "group_id"]) for s in counts}
    report = dict(root=str(root), counts=counts, sizes=dict(sizes), duplicates=duplicates,
                  class_pixels=class_counts.tolist(), class_fractions=(class_counts / class_counts.sum()).tolist(),
                  dataset_hash=dataset_hash, masks_roundtrip_checked=90 if strict_counts else int((table.source_split != "test").sum()),
                  official_train_valid_group_overlap=sorted(groups["train"] & groups["valid"]),
                  group_note="A-prefix is a filename group, not a confirmed physical specimen ID")
    write_json(out_dir / "audit.json", report)
    write_json(out_dir / "palette.json", dict(classes=["primary_si", "al3ni", "eutectic_si", "al"], colors=PALETTE.tolist()))
    return report


def verify_manifest(manifest):
    table = pd.read_csv(manifest).fillna("")
    for row in table.to_dict("records"):
        for kind in ("image", "mask"):
            if row[f"{kind}_path"] and sha256(row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                raise ValueError(f"Dataset changed: {row[f'{kind}_path']}; rerun prepare")
    return json_hash([(r["image_id"], r["source_split"], r["image_sha256"], r["mask_sha256"]) for r in table.to_dict("records")])
