from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset

from .common import PALETTE


@lru_cache(maxsize=16)
def gray(path):
    with Image.open(path) as image:
        if "A" in image.getbands() and np.any(np.asarray(image.getchannel("A")) != 255):
            raise ValueError(f"Transparent original: {path}")
        return np.asarray(image.convert("L"), dtype=np.uint8)


@lru_cache(maxsize=16)
def labels(path):
    with Image.open(path) as image:
        if "A" in image.getbands() and np.any(np.asarray(image.getchannel("A")) != 255):
            raise ValueError(f"Transparent target: {path}")
        rgb = np.asarray(image.convert("RGB"))
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for c, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = c
    if (out == 255).any():
        raise ValueError(f"Unknown mask palette: {path}")
    return out


def boundary_band(target, radius):
    edge = np.zeros(target.shape, dtype=bool)
    edge[1:] |= target[1:] != target[:-1]
    edge[:-1] |= target[1:] != target[:-1]
    edge[:, 1:] |= target[:, 1:] != target[:, :-1]
    edge[:, :-1] |= target[:, 1:] != target[:, :-1]
    return ndimage.distance_transform_edt(~edge) <= radius if edge.any() else edge


def crop(array, x, y, width, height, fill=None):
    h, w = array.shape[-2:]
    left, top = max(-x, 0), max(-y, 0)
    right, bottom = max(x + width - w, 0), max(y + height - h, 0)
    pads = [(0, 0)] * (array.ndim - 2) + [(top, bottom), (left, right)]
    padded = (np.pad(array, pads, mode="constant", constant_values=fill) if fill is not None
              else np.pad(array, pads, mode="reflect" if min(h, w) > 1 else "edge"))
    return padded[..., y + top:y + top + height, x + left:x + left + width].copy()


class Patches(Dataset):
    def __init__(self, rows, cfg, task, updates, sample_offset=0):
        if task not in ("ssl", "seg") or not rows or any(r["source_split"] != "train" for r in rows):
            raise ValueError("Training datasets accept train rows only")
        # The SSL dataset does not even retain mask paths or labels.
        self.rows = [{k: r[k] for k in ("image_id", "image_path")} for r in rows] if task == "ssl" else rows
        self.cfg, self.task, self.updates, self.offset = cfg, task, updates, sample_offset

    def __len__(self):
        return self.updates * self.cfg["train"]["effective_batch"]

    def __getitem__(self, index):
        cfg = self.cfg
        rng = np.random.default_rng(cfg["seed"] + int(index) + self.offset)
        row = self.rows[int(rng.integers(len(self.rows)))]
        g = gray(row["image_path"])
        target = labels(row["mask_path"]) if self.task == "seg" else None
        k = int(rng.integers(8))
        g = np.rot90(g, k % 4)
        if target is not None:
            target = np.rot90(target, k % 4)
        if k >= 4:
            g = g[:, ::-1]
            if target is not None:
                target = target[:, ::-1]
        core, halo = cfg["sampling"]["core"], cfg["sampling"]["halo"]
        size = core + 2 * halo
        x, y = int(rng.integers(g.shape[1])) - core // 2, int(rng.integers(g.shape[0])) - core // 2
        clean = crop(g, x - halo, y - halo, size, size).astype(np.float32) / 255
        center = (slice(halo, halo + core), slice(halo, halo + core))
        valid = np.zeros((size, size), np.float32)
        valid[center] = crop(np.ones(g.shape, np.float32), x, y, core, core, fill=0)
        if self.task == "seg":
            truth = np.full((size, size), 255, np.uint8)
            truth[center] = crop(target, x, y, core, core, fill=255)
            return torch.from_numpy(clean[None].repeat(3, 0)), torch.from_numpy(truth.astype(np.int64)), torch.from_numpy(valid)
        block = cfg["ssl"]["mask_block"]
        grid_size = size // block
        hidden = np.zeros(grid_size * grid_size, dtype=np.float32)
        count = max(1, min(hidden.size - 1, round(hidden.size * cfg["ssl"]["mask_ratio"])))
        hidden[rng.choice(hidden.size, count, replace=False)] = 1
        hidden = hidden.reshape(grid_size, grid_size).repeat(block, 0).repeat(block, 1)
        if not (hidden * valid).any():
            yy, xx = np.argwhere(valid > 0)[0]
            hidden[(yy // block)*block:(yy // block+1)*block, (xx // block)*block:(xx // block+1)*block] = 1
        corrupted = clean.copy()
        corrupted[hidden > 0] = cfg["ssl"]["fill"]
        return (torch.from_numpy(corrupted[None].repeat(3, 0)), torch.from_numpy(clean[None]),
                torch.from_numpy((hidden * valid)[None]))
