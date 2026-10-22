from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import Dataset

from .common import PALETTE


@lru_cache(maxsize=100)
def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.


@lru_cache(maxsize=90)
def labels(path):
    with Image.open(path) as im:
        if "A" in im.getbands() and np.any(np.asarray(im.getchannel("A")) != 255):
            raise ValueError(f"Transparent target: {path}")
        rgb = np.asarray(im.convert("RGB"))
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for c, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = c
    if (out == 255).any():
        raise ValueError(f"Unknown palette color: {path}")
    return out


def boundary(target):
    valid = target != 255
    edge = np.zeros(target.shape, bool)
    delta = (target[1:] != target[:-1]) & valid[1:] & valid[:-1]
    edge[1:] |= delta
    edge[:-1] |= delta
    delta = (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, 1:] |= delta
    edge[:, :-1] |= delta
    return edge


def boundary_band(target, radius):
    edge = boundary(target)
    return (ndimage.distance_transform_edt(~edge) <= radius) if edge.any() else edge


def crop(array, x, y, width, height, fill=None):
    h, w = array.shape[-2:]
    left, top = max(-x, 0), max(-y, 0)
    right, bottom = max(x + width - w, 0), max(y + height - h, 0)
    pads = [(0, 0)] * (array.ndim - 2) + [(top, bottom), (left, right)]
    padded = (np.pad(array, pads, mode="constant", constant_values=fill) if fill is not None
              else np.pad(array, pads, mode="reflect" if min(h, w) > 1 else "edge"))
    return padded[..., y + top:y + top + height, x + left:x + left + width].copy()


class PatchDataset(Dataset):
    """Index-based RNG keeps crop/augmentation identical across resolution experiments."""
    def __init__(self, rows, cfg):
        self.rows, self.cfg = rows, cfg

    def __len__(self):
        return self.cfg["train"]["updates"] * self.cfg["train"]["effective_batch"]

    def __getitem__(self, index):
        cfg = self.cfg
        rng = np.random.default_rng(cfg["seed"] + int(index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        g, t = gray(row["image_path"]).copy(), labels(row["mask_path"]).copy()
        aug = cfg["augmentation"]
        if aug["d4"]:
            k = int(rng.integers(8))
            g, t = np.rot90(g, k % 4), np.rot90(t, k % 4)
            if k >= 4:
                g, t = g[:, ::-1], t[:, ::-1]
        if rng.random() < aug["photometric_probability"]:
            g = np.clip((g - .5) * rng.uniform(*aug["contrast"]) + .5 +
                        rng.uniform(-aug["brightness"], aug["brightness"]), 0, 1)
            g = np.power(g, rng.uniform(*aug["gamma"]))
        core, halo, scale = cfg["sampling"]["core"], cfg["sampling"]["halo"], cfg["scale"]
        cy, cx = int(rng.integers(g.shape[0])), int(rng.integers(g.shape[1]))
        x, y = cx - core // 2, cy - core // 2
        edge = boundary_band(t, cfg["loss"]["boundary_radius"])
        g = crop(g, x - halo, y - halo, core + 2 * halo, core + 2 * halo)
        mask = np.full(g.shape, 255, np.uint8)
        weights = np.zeros(g.shape, np.float32)
        center = (slice(halo, halo + core), slice(halo, halo + core))
        mask[center] = crop(t, x, y, core, core, fill=255)
        weights[center] = crop(edge.astype(np.float32), x, y, core, core, fill=0)
        # The boundary band was generated in native pixels, before enlargement.
        size = ((core + 2 * halo) * scale,) * 2
        image = F.interpolate(torch.from_numpy(g.copy())[None, None], size=size,
                              mode="bilinear", align_corners=False)[0].expand(3, -1, -1).contiguous()
        target = F.interpolate(torch.from_numpy(mask)[None, None].float(), size=size, mode="nearest")[0, 0].long()
        edge = F.interpolate(torch.from_numpy(weights)[None, None], size=size, mode="nearest")[0, 0]
        return image, target, edge


class UpdateIndices:
    def __init__(self, length, rank, world, start):
        self.items = range(start + rank, length, world)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)
