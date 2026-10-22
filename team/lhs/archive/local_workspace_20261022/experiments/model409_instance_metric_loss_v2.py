#!/usr/bin/env python3
"""Memory-bounded Model409 instance-local feature contrast (revision 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import model409_instance_metric_loss as base


ComponentPair = base.ComponentPair
ComponentPairs = base.ComponentPairs
ProjectionHead = base.ProjectionHead
build_component_pairs = base.build_component_pairs
TEMPERATURE = base.TEMPERATURE


def _sample_many(
    projected_float: torch.Tensor,
    batch_index: int,
    flat_indices: torch.Tensor,
    mask_height: int,
    mask_width: int,
) -> torch.Tensor:
    """Sample all coordinates for one image in a single grid_sample call."""
    _, channels, _, _ = projected_float.shape
    indices = flat_indices.to(device=projected_float.device, non_blocking=True)
    y = torch.div(indices, mask_width, rounding_mode="floor").float()
    x = (indices % mask_width).float()
    grid = torch.stack((2.0 * (x + 0.5) / mask_width - 1.0,
                        2.0 * (y + 0.5) / mask_height - 1.0), dim=-1)
    grid = grid.reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        projected_float[batch_index:batch_index + 1], grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return F.normalize(sampled.reshape(channels, -1).transpose(0, 1), dim=1)


def component_local_metric_loss(
    projected: torch.Tensor,
    targets: ComponentPairs,
    *,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """Contrast same-component positives with nearby, labeled confusers.

    Unlike v1, projected maps are cast once and sampled once per image. This
    avoids retaining one full feature-map copy and grid-sample backward context
    per connected component, which caused activation memory to grow with the
    number of tiny regions in a crop.
    """
    if projected.ndim != 4 or temperature != TEMPERATURE:
        raise ValueError("Model409 requires BxDxHxW features and fixed T=0.2")
    if targets.batch_size != projected.shape[0] or not targets.pairs:
        if targets.batch_size != projected.shape[0]:
            raise ValueError("component metadata batch differs from projected features")
        return projected.float().sum() * 0.0
    projected_float = projected.float()
    losses_by_cell: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    pairs_by_batch: dict[int, list[ComponentPair]] = {}
    for pair in targets.pairs:
        pairs_by_batch.setdefault(pair.batch_index, []).append(pair)

    for batch_index, pairs in pairs_by_batch.items():
        all_indices: list[torch.Tensor] = []
        slices: list[tuple[ComponentPair, int, int, int, int]] = []
        cursor = 0
        for pair in pairs:
            n_positive = int(pair.positive_flat_indices.numel())
            n_negative = int(pair.negative_flat_indices.numel())
            all_indices.extend((pair.positive_flat_indices, pair.negative_flat_indices))
            slices.append((pair, cursor, cursor + n_positive,
                           cursor + n_positive, cursor + n_positive + n_negative))
            cursor += n_positive + n_negative
        sampled = _sample_many(
            projected_float,
            batch_index,
            torch.cat(all_indices),
            pairs[0].height,
            pairs[0].width,
        )
        for pair, pos_start, pos_end, neg_start, neg_end in slices:
            positive = sampled[pos_start:pos_end]
            negative = sampled[neg_start:neg_end]
            if positive.shape[0] < 2 or negative.shape[0] < 1:
                continue
            positive_logits = positive @ positive.T / temperature
            diagonal = torch.eye(positive.shape[0], device=positive.device, dtype=torch.bool)
            positive_logits = positive_logits.masked_fill(diagonal, -torch.inf)
            negative_logits = positive @ negative.T / temperature
            numerator = torch.logsumexp(positive_logits, dim=1)
            denominator = torch.logsumexp(
                torch.cat((positive_logits, negative_logits), dim=1), dim=1
            )
            cell = (pair.batch_index, pair.class_id, pair.bin_index)
            losses_by_cell.setdefault(cell, []).append((denominator - numerator).mean())
    if not losses_by_cell:
        return projected.float().sum() * 0.0
    return torch.stack([
        torch.stack(cell_values).mean() for cell_values in losses_by_cell.values()
    ]).mean()


def self_test() -> None:
    masks = torch.zeros((1, 512, 512), dtype=torch.long)
    # 392 disjoint 2x2 components: deliberately stress a dense microstructure crop.
    for row in range(14):
        for col in range(28):
            y, x = 8 + row * 12, 8 + col * 12
            masks[0, y:y + 2, x:x + 2] = 3
    targets = build_component_pairs(masks)
    assert targets.count == 392, targets.count
    projected = torch.randn(1, 8, 128, 128, requires_grad=True)
    loss = component_local_metric_loss(projected, targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert projected.grad is not None and torch.isfinite(projected.grad).all()
    print(f"Model409 v2 dense-component stress passed: pairs={targets.count}, loss={loss.item():.5f}", flush=True)


if __name__ == "__main__":
    self_test()
