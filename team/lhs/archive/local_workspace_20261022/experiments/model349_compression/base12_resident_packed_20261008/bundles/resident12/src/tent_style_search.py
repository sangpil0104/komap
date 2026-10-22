# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

import torch

from encoder_structure_search import build_model

from improve_five_causes import normalize_image

from improve_five_causes import scaled_pair

from improve_five_causes import tile_starts

_pristine_states = {}
_resident_model = None
_resident_device = None
_restores = 0

def load_aspp(path, device):
    global _resident_model, _resident_device, _restores
    key = str(path)
    if key not in _pristine_states:
        _pristine_states[key] = torch.load(path, map_location="cpu", weights_only=False)["model"]
    if _resident_model is None or _resident_device != str(device):
        _resident_model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
        _resident_device = str(device)
    # Restore every parameter and buffer before this image's fresh optimizer.
    _resident_model.load_state_dict(_pristine_states[key], strict=True)
    for parameter in _resident_model.parameters():
        parameter.grad = None
        parameter.requires_grad_(True)
    _resident_model.eval()
    _restores += 1
    print(f"RESIDENT source={key} restore={_restores} cpu_sources={len(_pristine_states)}", flush=True)
    return _resident_model

def adapt_model(checkpoint, raw, device, use_amp, steps, lr):
    model = load_aspp(checkpoint, device)
    if steps <= 0:
        return model
    scaled, _ = scaled_pair(raw, np.zeros(raw.shape[:2], dtype=np.int64), 768)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, 768, 576), tile_starts(width, 768, 576)
    locations = [(ys[0], xs[0]), (ys[0], xs[-1]), (ys[-1], xs[0]), (ys[-1], xs[-1])]
    locations = list(dict.fromkeys(locations))
    inputs = torch.stack([normalize_image(scaled[top:top + 768, left:left + 768])
                          for top, left in locations]).to(device)
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True); params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True); params.append(module.bias)
    # The ASPP image-pooling BN receives a 1x1 map; keep its source statistics.
    model.context.pool[2].eval()
    param_ids = {id(parameter) for parameter in params}
    for parameter in model.parameters():
        if id(parameter) not in param_ids:
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1)
            entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-7))).sum(1).mean()
        entropy.backward(); optimizer.step()
    model.eval()
    return model
