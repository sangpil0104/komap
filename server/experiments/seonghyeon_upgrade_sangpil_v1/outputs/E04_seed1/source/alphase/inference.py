import torch
from .contracts import relative_features, restore_native_probs
from .geometry import native_side, tile_positions, prepare_inputs, blend_weights
from .legacy import native_probs


@torch.inference_mode()
def predict_probs(model, gray, cfg, device):
    model.eval()
    if cfg["inference"]["mode"] == "legacy_full_frame":
        return native_probs(model, gray, cfg, device)
    h, w = gray.shape
    g = cfg["geometry"]
    inf = cfg["inference"]
    n = native_side(w, g["detail_input_size"], g["target_width"])
    image = torch.from_numpy(gray.copy()).to(device).float()[None, None] / 255
    rel = relative_features(image) if cfg["model"]["new_branches"]["texture"] else None
    total = torch.zeros(4, h, w, device=device, dtype=torch.float32)
    weights = torch.zeros(h, w, device=device, dtype=torch.float32)
    blend = blend_weights(n, device, inf["blend_weight_floor"])
    transforms = (
        [(k, flip) for flip in [False, True] for k in range(4)]
        if inf["d4"]
        else [(0, False)]
    )
    for y in tile_positions(h, n, inf["native_overlap_fraction"]):
        for x in tile_positions(w, n, inf["native_overlap_fraction"]):
            tile = torch.zeros(4, n, n, device=device, dtype=torch.float32)
            for k, flip in transforms:
                inputs = prepare_inputs(image, (x, y, n, n), cfg, k, flip, rel)
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.device(device).type == "cuda",
                ):
                    logits = model(**inputs)["logits_detail"]
                tile += restore_native_probs(
                    logits.float().softmax(1), (n, n), k, flip
                )[0] / len(transforms)
            oh = min(n, h - y)
            ow = min(n, w - x)
            total[:, y : y + oh, x : x + ow] += tile[:, :oh, :ow] * blend[:oh, :ow]
            weights[y : y + oh, x : x + ow] += blend[:oh, :ow]
    if not (weights > 0).all():
        raise RuntimeError("Tile coverage has zero weights")
    total /= weights
    return total / total.sum(0, keepdim=True)
