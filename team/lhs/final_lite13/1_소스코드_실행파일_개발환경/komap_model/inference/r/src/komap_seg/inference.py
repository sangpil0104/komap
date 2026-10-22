# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from typing import Callable

import torch

import torch.nn.functional as F

from komap_seg.palette import NUM_CLASSES

def _gaussian_window(size: int, sigma_scale: float = 0.125) -> torch.Tensor:
    sigma = size * sigma_scale
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g2d = g1d[:, None] * g1d[None, :]
    return (g2d / g2d.max()).clamp(min=1e-3)

def _tile_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts

@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int = 384,
    overlap: int = 128,
    num_classes: int = NUM_CLASSES,
    batch_size: int = 8,
    input_scale: float = 1.0,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    """Sliding-window softmax probability map at the image's original size.

    `image`: (3, H, W), already normalized the same way as training.
    `input_scale`: must match training's KomapPatchDataset(input_scale=...)
    -- tiles are cut at `patch_size / input_scale` native pixels, upscaled
    to `patch_size` for the model, then the output probabilities are
    downscaled back before stitching.
    Returns: (num_classes, H, W) on CPU.
    """
    _, h, w = image.shape
    tile_size = round(patch_size / input_scale)
    stride = tile_size - overlap

    pad_h = max(tile_size - h, 0)
    pad_w = max(tile_size - w, 0)
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    _, H, W = padded.shape

    ys = _tile_starts(H, tile_size, stride)
    xs = _tile_starts(W, tile_size, stride)
    tiles = [(y, x) for y in ys for x in xs]

    window = _gaussian_window(tile_size).to(device)
    padded = padded.to(device)

    prob_sum = torch.zeros(num_classes, H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    model.eval()
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        batch = torch.stack(
            [padded[:, y : y + tile_size, x : x + tile_size] for y, x in batch_tiles]
        )
        if input_scale != 1.0:
            batch = F.interpolate(batch, size=(patch_size, patch_size), mode="bicubic", align_corners=False)
        probs = torch.softmax(model(batch), dim=1)
        if input_scale != 1.0:
            probs = F.interpolate(probs, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        for (y, x), p in zip(batch_tiles, probs):
            prob_sum[:, y : y + tile_size, x : x + tile_size] += p * window
            weight_sum[y : y + tile_size, x : x + tile_size] += window

    prob_avg = prob_sum / weight_sum.clamp(min=1e-8)
    return prob_avg[:, :h, :w].cpu()

_TTA_OPS: list[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = [
    (lambda x: x, lambda x: x),
    (lambda x: x.flip(-1), lambda x: x.flip(-1)),
    (lambda x: x.flip(-2), lambda x: x.flip(-2)),
    (lambda x: x.flip(-1).flip(-2), lambda x: x.flip(-1).flip(-2)),
]

def _rotate90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(-2, -1))

_D4_OPS: list[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = [
    *[(lambda x, k=k: _rotate90(x, k),
       lambda x, k=k: _rotate90(x, -k)) for k in range(4)],
    *[(lambda x, k=k: _rotate90(x.flip(-1), k),
       lambda x, k=k: _rotate90(x, -k).flip(-1)) for k in range(4)],
]

@torch.no_grad()
def predict_probs_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int = 384,
    overlap: int = 128,
    num_classes: int = NUM_CLASSES,
    batch_size: int = 8,
    tta: bool = True,
    input_scale: float = 1.0,
    device: torch.device = torch.device("cuda:0"),
    tta_mode: str = "flip4",
) -> torch.Tensor:
    """Overlap-tile probabilities with optional flip4 or dihedral-8 TTA.

    Returns (num_classes, H, W) on CPU. Exposed separately from `predict()`
    (which just argmaxes this) so cross-fold ensembling (average several
    models' probability maps before argmax) can reuse it directly.
    """
    if tta_mode not in {"flip4", "d4"}:
        raise ValueError(f"unsupported TTA mode: {tta_mode}")
    if not tta:
        ops = _TTA_OPS[:1]
    else:
        ops = _TTA_OPS if tta_mode == "flip4" else _D4_OPS
    prob_sum = None
    for forward, inverse in ops:
        probs = predict_probs(
            model,
            forward(image),
            patch_size=patch_size,
            overlap=overlap,
            num_classes=num_classes,
            batch_size=batch_size,
            input_scale=input_scale,
            device=device,
        )
        probs = inverse(probs)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    return prob_sum / len(ops)
