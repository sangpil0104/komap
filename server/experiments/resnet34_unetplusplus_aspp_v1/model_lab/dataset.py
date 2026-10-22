from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PALETTE = np.asarray(
    [(76, 178, 76), (25, 76, 153), (204, 204, 204), (153, 127, 76)],
    dtype=np.uint8,
)


def load_image(path: str | Path) -> np.ndarray:
    """Load image from disk and return grayscale float32 array in [0, 1]."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def decode_mask(path: str | Path) -> np.ndarray:
    """Decode RGB/RGBA palette PNG into int64 [H, W] label array."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    labels = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = class_id
    if (labels == 255).any():
        unknown_colors = np.unique(rgb[labels == 255], axis=0)
        raise ValueError(f"Unknown mask colors in {path}: {unknown_colors.tolist()[:5]}")
    return labels


def save_mask(labels: np.ndarray, path: str | Path, rgba: bool = True) -> None:
    """Convert int64 label array [H, W] into RGB/RGBA PNG using competition PALETTE."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D label array, got shape {labels.shape}")
    clipped = np.clip(labels, 0, len(PALETTE) - 1)
    rgb = PALETTE[clipped]
    if rgba:
        alpha = np.full((*labels.shape, 1), 255, dtype=np.uint8)
        out_img = np.concatenate([rgb, alpha], axis=-1)
    else:
        out_img = rgb
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(p, format="PNG")


def compute_3channels(gray: np.ndarray) -> np.ndarray:
    """Computes standard 3-channel input: [raw gray, z-score normalized, empirical CDF]."""
    g = gray.astype(np.float32)
    # Channel 1: z-score
    mean_val = float(g.mean())
    std_val = max(float(g.std()), 1.0 / 255.0)
    z = np.clip((g - mean_val) / std_val, -5.0, 5.0) / 5.0

    # Channel 2: Empirical CDF
    bins = np.clip(np.rint(g * 255.0), 0, 255).astype(np.int64)
    counts = np.bincount(bins.ravel(), minlength=256)
    cdf = counts.cumsum().astype(np.float32) / float(bins.size)
    cdf_channel = cdf[bins]

    return np.stack([g, z, cdf_channel], axis=0)


def pad_to_multiple(array: np.ndarray, multiple: int = 32, fill: float | int | None = None) -> np.ndarray:
    """Pads trailing two dimensions (H, W) to multiples of `multiple`."""
    h, w = array.shape[-2:]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return array

    if fill is None:
        pad_width = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
        return np.pad(array, pad_width, mode="reflect")
    else:
        pad_width = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
        return np.pad(array, pad_width, mode="constant", constant_values=fill)


class KomapTrainDataset(Dataset):
    """Training dataset supporting patch-based sampling, D4 augmentation, and full-frame mode."""
    def __init__(
        self,
        samples: list[dict[str, Any]],
        cfg: dict[str, Any],
        stage_cfg: dict[str, Any],
        stage_index: int = 0,
    ) -> None:
        self.samples = samples
        self.cfg = cfg
        self.stage = stage_cfg
        self.seed = int(cfg.get("seed", 2026)) + stage_index * 1_000_000
        self.d4 = cfg.get("augmentation", {}).get("d4", True)
        self.photometric = cfg.get("augmentation", {}).get("photometric_probability", 0.5)

    def __len__(self) -> int:
        max_updates = self.stage.get("max_updates", 2000)
        eff_batch = self.cfg.get("train", {}).get("effective_batch_size", 8)
        return max_updates * eff_batch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(index))
        sample = self.samples[int(rng.integers(len(self.samples)))]

        # Load image & mask
        gray = load_image(sample["image_path"])
        mask = decode_mask(sample["mask_path"])
        h, w = gray.shape

        # Geometric D4 augmentation (8 dihedrals)
        if self.d4:
            k = int(rng.integers(8))
            rot_k = k % 4
            do_flip = (k // 4) == 1
            if rot_k > 0:
                gray = np.rot90(gray, rot_k)
                mask = np.rot90(mask, rot_k)
            if do_flip:
                gray = np.fliplr(gray)
                mask = np.fliplr(mask)
            h, w = gray.shape

        # Photometric augmentation
        if rng.random() < self.photometric:
            b_offset = rng.uniform(-0.04, 0.04)
            c_gain = rng.uniform(0.9, 1.1)
            gamma = rng.uniform(0.9, 1.1)
            gray = np.clip((gray - 0.5) * c_gain + 0.5 + b_offset, 0.0, 1.0)
            gray = np.power(np.maximum(gray, 1e-6), gamma)

        # Mode: full-frame vs patch
        mode = self.stage.get("mode", "patch")
        if mode == "full_frame":
            channels = compute_3channels(gray)
            padded_ch = pad_to_multiple(channels, multiple=32)
            padded_mask = pad_to_multiple(mask, multiple=32, fill=255)
            return {
                "image": torch.from_numpy(padded_ch.copy()),
                "mask": torch.from_numpy(padded_mask.copy()),
            }

        # Patch extraction
        core_size = int(self.stage.get("core_size", 512))
        halo = int(self.cfg.get("sampling", {}).get("halo", 64))
        patch_size = core_size + 2 * halo

        # Candidate sampling center
        mix = self.cfg.get("sampling", {}).get("mix", {"uniform": 0.6, "eutectic_center": 0.25, "error_or_boundary": 0.15})
        pick_mode = rng.random()
        candidates = None

        if pick_mode < mix.get("eutectic_center", 0.25):
            candidates = np.flatnonzero(mask == 2)  # Eutectic Si
        elif pick_mode < mix.get("eutectic_center", 0.25) + mix.get("error_or_boundary", 0.15):
            boundary = np.zeros(mask.shape, dtype=bool)
            boundary[1:] |= mask[1:] != mask[:-1]
            boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
            candidates = np.flatnonzero(boundary)

        if candidates is not None and len(candidates) > 0:
            cy, cx = np.unravel_index(rng.choice(candidates), mask.shape)
        else:
            cy = rng.integers(0, h)
            cx = rng.integers(0, w)

        # Compute crop coordinates
        half = patch_size // 2
        y1 = max(0, min(h - patch_size, cy - half)) if h >= patch_size else 0
        x1 = max(0, min(w - patch_size, cx - half)) if w >= patch_size else 0
        y2 = min(h, y1 + patch_size)
        x2 = min(w, x1 + patch_size)

        crop_g = gray[y1:y2, x1:x2]
        crop_m = mask[y1:y2, x1:x2]

        channels = compute_3channels(crop_g)
        padded_ch = pad_to_multiple(channels, multiple=32)
        padded_mask = pad_to_multiple(crop_m, multiple=32, fill=255)

        return {
            "image": torch.from_numpy(padded_ch.copy()),
            "mask": torch.from_numpy(padded_mask.copy()),
        }


class KomapEvalDataset(Dataset):
    """Validation or Test dataset delivering full-frame inputs and target masks."""
    def __init__(self, samples: list[dict[str, Any]], has_masks: bool = True) -> None:
        self.samples = samples
        self.has_masks = has_masks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        gray = load_image(sample["image_path"])
        h, w = gray.shape
        channels = compute_3channels(gray)
        padded_ch = pad_to_multiple(channels, multiple=32)

        data = {
            "image": torch.from_numpy(padded_ch),
            "original_h": h,
            "original_w": w,
            "image_id": sample["image_id"],
            "image_path": str(sample["image_path"]),
        }
        if self.has_masks and "mask_path" in sample and sample["mask_path"] is not None:
            mask = decode_mask(sample["mask_path"])
            data["mask"] = torch.from_numpy(mask)

        return data

