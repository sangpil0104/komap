"""Checkpoints bundle the weights with the `build_model` kwargs needed to
rebuild the same network, so evaluate.py/predict.py only need a path."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from komap_seg.model import build_model


def save_checkpoint(path: Path, model: nn.Module, model_config: dict[str, Any]) -> None:
    torch.save({"model_config": model_config, "state_dict": model.state_dict()}, path)


def load_model(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # encoder_weights=None: the checkpoint holds every weight, no need to
    # download the ImageNet encoder again.
    model = build_model(**ckpt["model_config"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
