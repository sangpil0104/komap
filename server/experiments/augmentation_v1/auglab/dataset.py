"""New training-only data adapter; model, loss, optimizer and evaluator stay frozen.

The base sampling logic is copied from the preserved TrainDataset. When extension
probabilities are zero this is bit-for-bit equivalent, including RNG consumption.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from alphase.data.dataset import TrainDataset, load_image, load_target, build_inputs, pad_array
from alphase.data.geometry import crop_array


def resize_pair(gray, target, scale):
    if scale == 1:
        return gray, target
    h, w = gray.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # F-mode preserves image precision. Categorical masks always use nearest.
    gray = np.asarray(Image.fromarray(np.ascontiguousarray(gray, dtype=np.float32)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32).copy()
    target = np.asarray(Image.fromarray(np.ascontiguousarray(target, dtype=np.uint8)).resize(
        size, Image.Resampling.NEAREST), dtype=np.uint8).copy()
    return gray, target


def transform(gray, target, rng, aug, extension, stage):
    """No in-place writes to cached images/masks; no mask photometric transforms."""
    g, target = gray, target.copy()
    if aug["d4"]:
        k = int(rng.integers(8))
        g, target = np.rot90(g, k % 4), np.rot90(target, k % 4)
        if k // 4:
            g, target = g[:, ::-1], target[:, ::-1]
    # Preserve the original final stage: native resolution, no new corruption.
    patch_stage = stage["mode"] != "full_frame"
    pscale = extension.get("scale_probability", 0.) if patch_stage else 0.
    if pscale and rng.random() < pscale:
        g, target = resize_pair(g, target, float(rng.choice(extension["scale_choices"])))
    probability = stage.get("photometric_probability", aug["photometric_probability"])
    if rng.random() < probability:
        g = np.clip((g-.5) * rng.uniform(*aug["contrast_gain"]) + .5 + rng.uniform(*aug["brightness_offset"]), 0, 1)
        g = np.power(g, rng.uniform(*aug["gamma"]))
    pnoise = extension.get("noise_probability", 0.) if patch_stage else 0.
    if pnoise and rng.random() < pnoise:
        std = rng.uniform(*extension["noise_std"])
        noise = rng.standard_normal(g.shape, dtype=np.float32) * std
        g = np.clip(g + noise, 0, 1)
    return np.ascontiguousarray(g, dtype=np.float32), np.ascontiguousarray(target)


class AugmentedDataset(TrainDataset):
    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        gray = load_image(row["image_path"]).astype(np.float32) / 255
        target = load_target(row["mask_path"])
        gray, target = transform(gray, target, rng, self.cfg["augmentation"],
                                 self.cfg.get("augmentation_extension", {}), self.stage)
        # Recompute gray/z-score/CDF after all intensity/geometric changes.
        detail, dino = build_inputs(gray, self.raw_only, self.hybrid)
        h, w = gray.shape
        if self.stage["mode"] == "full_frame":
            mask = pad_array(target, 255).astype(np.int64)
            box = (0, 0, detail.shape[-1], detail.shape[-2])
        else:
            core = int(rng.choice(self.stage["core_sizes"]))
            halo = self.cfg["sampling"]["halo"]
            if core % 32 or (core + 2 * halo) % 32:
                raise ValueError("Core and halo input dimensions must be multiples of 32")
            mode, candidates = rng.random(), None
            mix = self.cfg["sampling"]["mix"]
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
            x, y = int(cx) - core // 2, int(cy) - core // 2
            box = (x-halo, y-halo, x+core+halo, y+core+halo)
            detail = torch.from_numpy(crop_array(detail.numpy()[..., :h, :w], box))
            mask = np.full((core+2*halo, core+2*halo), 255, dtype=np.int64)
            mask[halo:halo+core, halo:halo+core] = crop_array(target, (x, y, x+core, y+core), 255)
        if not (mask != 255).any():
            raise ValueError("Sampler produced an empty target")
        return dict(detail=detail, dino=dino, target=torch.from_numpy(mask),
                    box=torch.tensor(box, dtype=torch.float32), image_id=row["image_id"])
