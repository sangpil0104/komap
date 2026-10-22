from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from alphase.data.geometry import crop_array
from alphase.data.palette import decode_mask


@lru_cache(maxsize=100)
def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


@lru_cache(maxsize=90)
def load_target(path):
    return decode_mask(path).astype(np.uint8)


def channels(gray, raw_only=False):
    """Compute statistics on the real, full, augmented image before any padding."""
    g = gray.astype(np.float32)
    if raw_only:
        return g[None]
    bins = np.clip(np.rint(g * 255), 0, 255).astype(np.uint8)
    cdf = np.bincount(bins.ravel(), minlength=256).cumsum().astype(np.float32) / bins.size
    z = np.clip((g-g.mean()) / max(float(g.std()), 1/255), -5, 5) / 5
    return np.stack([g, z, cdf[bins]])


def pad_array(array, fill=None):
    h, w = array.shape[-2:]
    return crop_array(array, (0, 0, w + (-w % 32), h + (-h % 32)), fill)


def build_inputs(gray, raw_only=False, hybrid=False):
    detail = torch.from_numpy(pad_array(channels(gray, raw_only)))
    if hybrid:
        dino = pad_array(np.repeat(gray[None], 3, axis=0))
        dino = (dino - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
        dino = torch.from_numpy(dino)
    else:
        dino = torch.empty(0)
    return detail, dino


class TrainDataset(Dataset):
    """An index defines its own RNG: reproducible across prefetch and resume."""
    def __init__(self, rows, cfg, stage, stage_index=0):
        self.rows, self.cfg, self.stage = rows, cfg, stage
        self.seed = int(cfg["seed"]) + stage_index * 100_000_000
        self.hybrid = cfg["model"]["name"] == "alphase_hybrid_v1"
        self.raw_only = cfg["data"]["image_channels"] == ["gray"]

    def __len__(self):
        return self.stage["max_updates"] * self.cfg["train"]["effective_batch_size"]

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        g = load_image(row["image_path"]).astype(np.float32) / 255
        target = load_target(row["mask_path"]).copy()
        aug = self.cfg["augmentation"]
        if aug["d4"]:
            k = int(rng.integers(8))
            g, target = np.rot90(g, k % 4), np.rot90(target, k % 4)
            if k // 4:
                g, target = g[:, ::-1], target[:, ::-1]
        probability = self.stage.get("photometric_probability", aug["photometric_probability"])
        if rng.random() < probability:
            g = np.clip((g-.5) * rng.uniform(*aug["contrast_gain"]) + .5 + rng.uniform(*aug["brightness_offset"]), 0, 1)
            g = np.power(g, rng.uniform(*aug["gamma"]))
        detail, dino = build_inputs(np.ascontiguousarray(g, dtype=np.float32), self.raw_only, self.hybrid)
        h, w = g.shape
        if self.stage["mode"] == "full_frame":
            mask = pad_array(target, 255).astype(np.int64)
            box = (0, 0, detail.shape[-1], detail.shape[-2])
        else:
            core = int(rng.choice(self.stage["core_sizes"]))
            halo = self.cfg["sampling"]["halo"]
            if core % 32 or (core+2*halo) % 32:
                raise ValueError("Core and halo input dimensions must be multiples of 32")
            mode = rng.random()
            mix = self.cfg["sampling"]["mix"]
            candidates = None
            if mode < mix["eutectic_center"]:
                candidates = np.flatnonzero(target == 2)
            elif mode < mix["eutectic_center"] + mix["error_or_boundary"]:
                boundary = np.zeros(target.shape, bool)
                boundary[1:] |= target[1:] != target[:-1]
                boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
                candidates = np.flatnonzero(boundary)
            if candidates is not None and len(candidates):
                cy, cx = np.unravel_index(rng.choice(candidates), target.shape)
            else:
                cy, cx = rng.integers(h), rng.integers(w)
            x, y = int(cx)-core//2, int(cy)-core//2
            box = (x-halo, y-halo, x+core+halo, y+core+halo)
            # Crop from the real image, not from already-reflected right/bottom padding.
            detail = torch.from_numpy(crop_array(detail.numpy()[..., :h, :w], box))
            mask = np.full((core+2*halo, core+2*halo), 255, dtype=np.int64)
            mask[halo:halo+core, halo:halo+core] = crop_array(target, (x, y, x+core, y+core), 255)
        if not (mask != 255).any():
            raise ValueError("Sampler produced an empty target")
        return dict(detail=detail, dino=dino, target=torch.from_numpy(mask),
                    box=torch.tensor(box, dtype=torch.float32), image_id=row["image_id"])


class RankIndices:
    def __init__(self, length, rank, world_size, start=0):
        self.indices = range(start+rank, length, world_size)
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
