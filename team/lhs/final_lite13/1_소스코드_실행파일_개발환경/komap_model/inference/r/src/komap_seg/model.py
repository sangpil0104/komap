# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from typing import Optional

import segmentation_models_pytorch as smp

import torch.nn as nn

from komap_seg.palette import NUM_CLASSES

def build_model(
    architecture: str = "UnetPlusPlus",
    encoder_name: str = "tu-hrnet_w48",
    encoder_weights: Optional[str] = "imagenet",
    boundary_head: bool = True,
    sdm_head: bool = False,
    **kwargs,
) -> nn.Module:
    """Any segmentation_models_pytorch architecture/encoder, 3-channel input,
    NUM_CLASSES output logits at input resolution.

    `boundary_head=True` adds `model.boundary_head`, a Conv2d(decoder_channels,
    1) predicting the class-agnostic boundary band (komap_seg/losses.py
    BoundaryHeadLoss). It is only used in training through
    `forward_with_boundary`; `model(x)` stays the plain smp forward returning
    segmentation logits, so inference never touches the extra head.

    `sdm_head` and extra `kwargs` (e.g. `output_stride`) are accepted so
    checkpoints from the original experiment code load unchanged; the
    released recipe doesn't use them.
    """
    if sdm_head:
        raise ValueError("sdm_head is not part of this release")
    model_cls = getattr(smp, architecture)
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        **kwargs,
    )
    if boundary_head:
        decoder_channels = model.segmentation_head[0].in_channels
        model.boundary_head = nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1)
    return model
