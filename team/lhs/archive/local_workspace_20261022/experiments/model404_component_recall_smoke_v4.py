#!/usr/bin/env python3
"""CPU synthetic checks for the balanced two-class Model404 v4 loss."""

from __future__ import annotations

import torch

from model404_component_recall_loss_v4 import (
    AREA_BINS,
    TARGET_CLASSES,
    build_component_targets,
    component_recall_loss,
)


def main() -> None:
    masks = torch.zeros((2, 96, 96), dtype=torch.long)
    masks[0, 10:12, 10:12] = 1                    # Al3Ni, 4 px, bin 0
    masks[0, 20:24, 20:24] = 1                    # Al3Ni, 16 px, bin 1
    masks[0, 35:43, 20:28] = 3                    # Eutectic, 64 px, bin 2
    masks[0, 50:54, 50:54] = 3                    # Eutectic, 16 px, bin 1
    masks[0, 0:3, 60:64] = 3                      # crop-truncated, ignored
    masks[0, 60:93, 60:93] = 1                    # >1024 px, ignored
    masks[1, 15:18, 15:18] = 1                    # Al3Ni, 9 px, bin 0
    masks[1, 40:44, 40:44] = 2                    # Primary Si, ignored

    targets = build_component_targets(masks)
    assert targets.count == 5, targets.count
    assert sorted(targets.areas.tolist()) == [4.0, 9.0, 16.0, 16.0, 64.0]
    assert sorted(targets.class_ids.tolist()) == [1, 1, 1, 3, 3]
    assert sorted(targets.class_slots.tolist()) == [0, 0, 0, 1, 1]
    assert sorted(targets.bin_indices.tolist()) == [0, 0, 1, 1, 2]

    logits = torch.zeros((2, 4, 96, 96), requires_grad=True)
    loss = component_recall_loss(logits, targets)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    for component_id, class_id in enumerate(targets.class_ids.tolist(), start=1):
        pixels = targets.component_ids == component_id
        assert (logits.grad[:, class_id][pixels] < 0).all()
    assert torch.count_nonzero(logits.grad[0, :, 0, 60]) == 0
    assert torch.count_nonzero(logits.grad[0, :, 75, 75]) == 0
    assert torch.count_nonzero(logits.grad[1, :, 41, 41]) == 0

    custom = build_component_targets(masks, area_bins=((4, 64), (64, 1025)))
    custom_loss = component_recall_loss(logits.detach().requires_grad_(True), custom)
    assert torch.isfinite(custom_loss) and custom.num_bins == 2
    assert TARGET_CLASSES == (1, 3) and len(AREA_BINS) == 3

    empty = build_component_targets(torch.zeros((1, 16, 16), dtype=torch.long))
    zero = component_recall_loss(torch.randn((1, 4, 16, 16), requires_grad=True), empty)
    assert zero.item() == 0.0 and zero.requires_grad
    print("PASS: Al3Ni/Eutectic bins, balanced cells, border/area filtering, gradients, empty fallback")


if __name__ == "__main__":
    main()
