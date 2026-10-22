#!/usr/bin/env python3
"""CPU smoke test for Model377's directed Al3Ni-suppression loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model336_hrnet_best"))
from komap_seg.losses import directional_unlikelihood_loss  # noqa: E402


def main() -> None:
    labels = torch.zeros((1, 3, 3), dtype=torch.long)
    labels[0, 1, 1] = 3  # target Eutectic Si pixel
    labels[0, 1, 2] = 1  # axial Al3Ni neighbor
    logits = torch.zeros((1, 4, 3, 3), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 1, 1] = 2.0

    loss, support = directional_unlikelihood_loss(logits, labels)
    probability = torch.softmax(logits.detach(), dim=1)[0, 1, 1, 1]
    expected = -torch.log1p(-probability) / labels.numel()
    assert int(support) == 1, f"expected only the Eutectic-side interface pixel, got {support}"
    assert torch.isfinite(loss) and torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert logits.grad is not None and logits.grad[0, 1, 1, 1] > 0

    no_support_logits = torch.randn((1, 4, 3, 3), requires_grad=True)
    no_support, no_support_count = directional_unlikelihood_loss(
        no_support_logits, torch.zeros((1, 3, 3), dtype=torch.long)
    )
    assert int(no_support_count) == 0 and float(no_support.detach()) == 0.0
    no_support.backward()
    assert torch.isfinite(no_support_logits.grad).all()

    saturated = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
    saturated[0, 1, 1, 1] = 1000.0
    stable_loss, stable_count = directional_unlikelihood_loss(saturated, labels)
    assert int(stable_count) == 1 and torch.isfinite(stable_loss)

    print("Model377 directional unlikelihood smoke passed")


if __name__ == "__main__":
    main()
