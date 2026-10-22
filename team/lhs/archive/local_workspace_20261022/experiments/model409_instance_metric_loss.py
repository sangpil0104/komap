#!/usr/bin/env python3
"""Component-local feature contrast for tiny Al3Ni/Eutectic-Si regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_CLASSES = (1, 3)
CONFUSERS = {1: (0,), 3: (0, 1)}
AREA_BINS = ((4, 16), (16, 64), (64, 256))
TEMPERATURE = 0.2
MAX_POSITIVE_PIXELS = 32
MAX_NEGATIVE_PIXELS = 64


@dataclass(frozen=True)
class ComponentPair:
    batch_index: int
    class_id: int
    bin_index: int
    height: int
    width: int
    positive_flat_indices: torch.Tensor
    negative_flat_indices: torch.Tensor


@dataclass(frozen=True)
class ComponentPairs:
    pairs: tuple[ComponentPair, ...]
    batch_size: int
    height: int
    width: int

    @property
    def count(self) -> int:
        return len(self.pairs)


class ProjectionHead(nn.Module):
    """Training-only pointwise projection; discarded for inference."""

    def __init__(self, channels: int, projection_dim: int = 32):
        super().__init__()
        hidden = min(128, max(32, channels // 2))
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, projection_dim, kernel_size=1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _fixed_subsample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def build_component_pairs(
    masks: torch.Tensor,
    *,
    min_area: int = AREA_BINS[0][0],
    max_area: int = AREA_BINS[-1][1],
    inner_radius: int = 1,
    outer_radius: int = 6,
) -> ComponentPairs:
    """Build train-mask-only component positives and nearby confuser negatives.

    Eligible 8-connected GT components have augmented-mask areas in [4,256),
    do not touch the crop edge, and have at least one fixed-confuser pixel in
    the 2..6-pixel annulus around the component. Pixel coordinates are retained
    at full mask resolution and sampled from the decoder feature map later.
    """
    if masks.device.type != "cpu" or masks.ndim != 3:
        raise ValueError("component pairs require a CPU BxHxW training mask")
    if (min_area != AREA_BINS[0][0] or max_area != AREA_BINS[-1][1]
            or inner_radius != 1 or outer_radius != 6):
        raise ValueError("Model409 uses the fixed [4,256) area and 2..6px annulus")
    array = masks.detach().numpy()
    batch_size, height, width = array.shape
    ellipse_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ellipse_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    pairs: list[ComponentPair] = []

    for batch_index in range(batch_size):
        mask = np.ascontiguousarray(array[batch_index])
        for class_id in TARGET_CLASSES:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                np.ascontiguousarray(mask == class_id, dtype=np.uint8),
                connectivity=8,
                ltype=cv2.CV_32S,
            )
            for local_id in range(1, count):
                x, y, component_width, component_height, area = map(int, stats[local_id])
                if not (min_area <= area < max_area):
                    continue
                if (x == 0 or y == 0 or x + component_width >= width
                        or y + component_height >= height):
                    continue
                component = np.ascontiguousarray(labels == local_id, dtype=np.uint8)
                inner = cv2.dilate(component, ellipse_inner, iterations=1).astype(bool)
                outer = cv2.dilate(component, ellipse_outer, iterations=1).astype(bool)
                annulus = outer & ~inner
                confuser_mask = np.isin(mask, CONFUSERS[class_id]) & annulus
                positive = np.flatnonzero(component)
                negative = np.flatnonzero(confuser_mask)
                if positive.size < 2 or negative.size == 0:
                    continue
                bin_index = next(
                    index for index, (lower, upper) in enumerate(AREA_BINS)
                    if lower <= area < upper
                )
                positive = _fixed_subsample(positive, MAX_POSITIVE_PIXELS)
                negative = _fixed_subsample(negative, MAX_NEGATIVE_PIXELS)
                pairs.append(ComponentPair(
                    batch_index=batch_index,
                    class_id=class_id,
                    bin_index=bin_index,
                    height=height,
                    width=width,
                    positive_flat_indices=torch.from_numpy(positive.copy()).long(),
                    negative_flat_indices=torch.from_numpy(negative.copy()).long(),
                ))
    return ComponentPairs(tuple(pairs), batch_size, height, width)


def _sample_projected(
    projected: torch.Tensor,
    batch_index: int,
    flat_indices: torch.Tensor,
    mask_height: int,
    mask_width: int,
) -> torch.Tensor:
    _, channels, _, _ = projected.shape
    indices = flat_indices.to(device=projected.device, non_blocking=True)
    y = torch.div(indices, mask_width, rounding_mode="floor").float()
    x = (indices % mask_width).float()
    grid = torch.stack((2.0 * (x + 0.5) / mask_width - 1.0,
                        2.0 * (y + 0.5) / mask_height - 1.0), dim=-1)
    grid = grid.reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        projected[batch_index:batch_index + 1].float(), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return F.normalize(sampled.reshape(channels, -1).transpose(0, 1), dim=1)


def component_local_metric_loss(
    projected: torch.Tensor,
    targets: ComponentPairs,
    *,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """Instance-local supervised contrastive loss on decoder embeddings.

    For each tiny GT component, target pixels are positives only to other
    pixels of that same connected component. Negatives are only nearby pixels
    of the pre-registered confuser classes. Component losses are averaged
    equally within crop x class x size-bin cells. No logits, predictions,
    validation labels, or inference-time components enter the objective.
    """
    if projected.ndim != 4 or temperature != TEMPERATURE:
        raise ValueError("Model409 requires BxDxHxW projected features and fixed T=0.2")
    if (targets.batch_size != projected.shape[0]
            or targets.height <= 0 or targets.width <= 0):
        raise ValueError("component metadata does not match projected batch")
    if not targets.pairs:
        return projected.float().sum() * 0.0

    cell_losses: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    for pair in targets.pairs:
        positive = _sample_projected(
            projected, pair.batch_index, pair.positive_flat_indices,
            pair.height, pair.width,
        )
        negative = _sample_projected(
            projected, pair.batch_index, pair.negative_flat_indices,
            pair.height, pair.width,
        )
        if positive.shape[0] < 2 or negative.shape[0] < 1:
            continue
        similarity_positive = positive @ positive.T / temperature
        diagonal = torch.eye(
            positive.shape[0], device=positive.device, dtype=torch.bool
        )
        similarity_positive = similarity_positive.masked_fill(diagonal, -torch.inf)
        similarity_negative = positive @ negative.T / temperature
        numerator = torch.logsumexp(similarity_positive, dim=1)
        denominator = torch.logsumexp(
            torch.cat((similarity_positive, similarity_negative), dim=1), dim=1
        )
        loss = (denominator - numerator).mean()
        key = (pair.batch_index, pair.class_id, pair.bin_index)
        cell_losses.setdefault(key, []).append(loss)
    if not cell_losses:
        return projected.float().sum() * 0.0
    return torch.stack([torch.stack(values).mean() for values in cell_losses.values()]).mean()


def self_test() -> None:
    masks = torch.zeros((1, 64, 64), dtype=torch.long)
    masks[0, 24:27, 24:27] = 1
    masks[0, 36:40, 36:40] = 3
    targets = build_component_pairs(masks)
    assert targets.count == 2, targets.count
    projected = torch.randn(1, 8, 16, 16, requires_grad=True)
    loss = component_local_metric_loss(projected, targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert projected.grad is not None and torch.isfinite(projected.grad).all()
    print("Model409 local component metric loss smoke passed", flush=True)


if __name__ == "__main__":
    self_test()
