"""Checkpoints bundle the weights with the `build_model` kwargs needed to
rebuild the same network, so evaluate.py/predict.py only need a path."""

from pathlib import Path
import os
from typing import Any

import torch
import torch.nn as nn

from komap_seg.model import build_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    preprocessing_config: dict[str, Any] | None = None,
) -> None:
    checkpoint = {"model_config": model_config, "state_dict": model.state_dict()}
    if preprocessing_config is not None:
        checkpoint["preprocessing_config"] = preprocessing_config
    torch.save(checkpoint, path)


def resolve_preprocessing(
    saved: dict[str, Any] | None,
    patch_size: int | None = None,
    input_scale: float | None = None,
    target_width: int | None = None,
) -> dict[str, Any]:
    defaults = {"patch_size": 512, "input_scale": 2.0, "target_width": 0,
                "normalize_width": False}
    requested = {
        "patch_size": patch_size,
        "input_scale": input_scale,
        "target_width": target_width,
    }
    if saved is None:
        if target_width not in (None, 0):
            raise ValueError("checkpoint has no preprocessing metadata; width normalization cannot be verified")
        resolved = dict(defaults)
        for key, value in requested.items():
            if value is not None:
                resolved[key] = value
    else:
        required = set(defaults)
        if set(saved) not in (required, required | {"input_mode"}):
            raise ValueError(f"checkpoint preprocessing metadata has unexpected keys: {sorted(saved)}")
        resolved = dict(saved)
        if saved.get("input_mode","original") != os.environ.get("KOMAP_HR_INPUT_MODE","original"):
            raise ValueError("Input mode differs from checkpoint")
        for key, value in requested.items():
            if value is not None and value != saved[key]:
                raise ValueError(
                    f"requested {key}={value!r} differs from checkpoint preprocessing "
                    f"{saved[key]!r}"
                )
    if resolved["patch_size"] <= 0 or resolved["input_scale"] <= 0 or resolved["target_width"] < 0:
        raise ValueError(f"invalid checkpoint preprocessing values: {resolved}")
    if bool(resolved["normalize_width"]) != (resolved["target_width"] > 0):
        raise ValueError(f"width-normalization flag/target mismatch: {resolved}")
    return resolved


def load_model_and_preprocessing(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any] | None]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # encoder_weights=None: the checkpoint holds every weight, no need to
    # download the ImageNet encoder again.
    model = build_model(**ckpt["model_config"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt.get("preprocessing_config")


def load_model(path: Path, device: torch.device) -> nn.Module:
    model, _ = load_model_and_preprocessing(path, device)
    return model
