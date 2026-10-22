"""Small CPU-only smoke test for the fixed Model398 B4 tiled predictor."""

from __future__ import annotations

import numpy as np
import torch

from model398_b4_tiled_inference import (
    OVERLAP,
    PATCH_SIZE,
    TTA_MODE,
    prepare_b4_image,
    predict_b4_tiled_probs,
)


class TinyFourClassModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 0]
        return torch.stack((z, -z, 0.5 * z, -0.5 * z), dim=1)


def main() -> None:
    rng = np.random.default_rng(398)
    # Keep the resized grid at least one 1536px tile in both dimensions so
    # the production reflection-padding path is exercised without a test-only
    # override of the frozen inference settings.
    gray = rng.integers(0, 256, size=(1152, 864), dtype=np.uint8)
    target_width = 1536

    resized, normalized = prepare_b4_image(gray, target_width)
    if resized.shape != (2048, 1536):
        raise AssertionError(f"unexpected resized shape: {resized.shape}")

    # Verify the adapter's inputs are exactly Model329's whole-image pipeline.
    from model398_b4_tiled_inference import _B4_COMMON

    expected = _B4_COMMON.make_input(
        resized, norm="image_global", input_mode="zcdf3"
    )
    if not np.array_equal(normalized.numpy(), expected):
        raise AssertionError("B4 preprocessing differs from Model329 make_input")

    model = TinyFourClassModel().eval()
    probs = predict_b4_tiled_probs(
        model, gray, target_width=target_width, device="cpu"
    )
    if probs.shape != (4, *gray.shape):
        raise AssertionError(f"unexpected native-grid probability shape: {probs.shape}")
    if not np.isfinite(probs).all() or not np.allclose(
        probs.sum(axis=0), 1.0, atol=2e-3, rtol=0
    ):
        raise AssertionError("invalid probability values")
    if PATCH_SIZE != 1536 or OVERLAP != 384 or TTA_MODE != "d4":
        raise AssertionError("fixed Model398 inference settings changed")

    print("Model398 B4 tiled inference smoke passed (CPU, synthetic only).")


if __name__ == "__main__":
    main()
