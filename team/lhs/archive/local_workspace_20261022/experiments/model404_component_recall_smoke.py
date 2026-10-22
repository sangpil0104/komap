#!/usr/bin/env python3
"""CPU synthetic checks for Model404 component-recall loss."""

from __future__ import annotations

import torch

from model404_component_recall_loss import AREA_BINS, build_component_targets, component_recall_loss


def main() -> None:
    masks = torch.zeros((2, 80, 80), dtype=torch.long)
    masks[0, 10:12, 10:12] = 3                    # 4 px, bin 0
    masks[0, 20:24, 20:24] = 3                    # 16 px, bin 1
    masks[0, 35:43, 20:28] = 3                    # 64 px, bin 2
    masks[0, 0:3, 40:44] = 3                      # crop-truncated, ignored
    masks[0, 50:84, 40:74] = 3                    # >1024 px, ignored
    masks[1, 15:19, 15:19] = 3                    # 16 px, bin 1
    masks[1, 50:54, 50:54] = 2                    # other class, ignored

    targets = build_component_targets(masks)
    assert targets.count == 4, targets.count
    assert sorted(targets.areas.tolist()) == [4.0, 16.0, 16.0, 64.0]
    assert sorted(targets.bin_indices.tolist()) == [0, 1, 1, 2]
    assert sorted(targets.batch_indices.tolist()) == [0, 0, 0, 1]

    logits = torch.zeros((2, 4, 80, 80), requires_grad=True)
    loss = component_recall_loss(logits, targets)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # The target-class gradient must encourage increasing class-3 probability
    # on eligible component pixels; ignored pixels have no component-loss grad.
    eligible = targets.component_ids > 0
    assert (logits.grad[:, 3][eligible] < 0).all()
    assert torch.count_nonzero(logits.grad[0, :, 0, 40]) == 0
    assert torch.count_nonzero(logits.grad[0, :, 70, 50]) == 0
    assert torch.count_nonzero(logits.grad[1, :, 51, 51]) == 0

    custom = build_component_targets(masks, area_bins=((4, 64), (64, 1025)))
    custom_loss = component_recall_loss(logits.detach().requires_grad_(True), custom)
    assert torch.isfinite(custom_loss) and custom.num_bins == 2
    assert len(AREA_BINS) == 3

    empty = build_component_targets(torch.zeros((1, 16, 16), dtype=torch.long))
    zero = component_recall_loss(torch.randn((1, 4, 16, 16), requires_grad=True), empty)
    assert zero.item() == 0.0 and zero.requires_grad
    print("PASS: component bins, border/area filtering, finite gradients, and empty-batch fallback")


if __name__ == "__main__":
    main()
