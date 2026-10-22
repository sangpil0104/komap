# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


from pathlib import Path

import torch

import torch.nn as nn

from komap_seg.model import build_model

def load_model(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # encoder_weights=None: the checkpoint holds every weight, no need to
    # download the ImageNet encoder again.
    model = build_model(**ckpt["model_config"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
