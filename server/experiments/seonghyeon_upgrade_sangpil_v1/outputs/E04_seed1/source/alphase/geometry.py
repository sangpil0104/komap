"""Native pixel-edge rectangles and shared D4. No ground truth in inference."""

import math
import torch
import torch.nn.functional as F
from .contracts import d4_forward, d4_inverse, relative_features


def native_side(width, detail_size=1536, target_width=4000):
    return max(1, math.floor(detail_size * width / target_width + 0.5))


def context_rect(d, factor=2.0):
    x, y, w, h = d
    return (x + (1 - factor) * w / 2, y + (1 - factor) * h / 2, w * factor, h * factor)


def crop_resize(x, rect, out_hw, mode="bilinear", padding="reflection"):
    """Sample native edge coordinates at output pixel centers, align_corners=False."""
    if x.ndim != 4:
        raise ValueError("Expected BCHW")
    h, w = x.shape[-2:]
    oh, ow = out_hw
    x0, y0, rw, rh = rect
    xx = (torch.arange(ow, device=x.device, dtype=torch.float32) + 0.5) * rw / ow + x0
    yy = (torch.arange(oh, device=x.device, dtype=torch.float32) + 0.5) * rh / oh + y0
    gy, gx = torch.meshgrid(2 * yy / h - 1, 2 * xx / w - 1, indexing="ij")
    grid = torch.stack((gx, gy), -1)[None].expand(x.shape[0], -1, -1, -1)
    if min(h, w) == 1 and padding == "reflection":
        padding = "border"
    return F.grid_sample(
        x.float(), grid, mode=mode, padding_mode=padding, align_corners=False
    )


def native_target(target, rect):
    x, y, w, h = map(int, rect)
    yy = torch.arange(y, y + h, device=target.device)
    xx = torch.arange(x, x + w, device=target.device)
    valid = (
        (yy[:, None] >= 0)
        & (yy[:, None] < target.shape[-2])
        & (xx[None, :] >= 0)
        & (xx[None, :] < target.shape[-1])
    )
    out = target[
        yy.clamp(0, target.shape[-2] - 1)[:, None],
        xx.clamp(0, target.shape[-1] - 1)[None, :],
    ].clone()
    out[~valid] = 255
    return out, valid


def prepare_inputs(gray, d, cfg, k=0, flip=False, features=None):
    """gray is 1x1xHxW in [0,1]; D4 applied after consistent native crops."""
    g = cfg["geometry"]
    b = cfg["model"]["new_branches"]
    p = g["detail_input_size"]
    detail = crop_resize(gray, d, (p, p))
    result = {
        "detail_rgb": d4_forward(
            ((detail - 0.449) / 0.226).expand(-1, 3, -1, -1), k, flip
        ),
        "detail_rects": torch.tensor([d], device=gray.device, dtype=torch.float32),
        "d4": [(k, flip)],
    }
    if b["texture"]:
        if features is None:
            r = cfg["relative_features"]
            features = relative_features(
                gray,
                tuple(r["sigma_native_px"]),
                r["eps_std"],
                r["clip_z"],
                r["gaussian_truncate"],
            )
        result["texture_detail"] = d4_forward(crop_resize(features, d, (p, p)), k, flip)
    if b["context"]:
        c = context_rect(d, g["context_factor"])
        size = g["context_input_size"]
        result["context_gray"] = d4_forward(crop_resize(gray, c, (size, size)), k, flip)
        result["context_rects"] = torch.tensor(
            [c], device=gray.device, dtype=torch.float32
        )
    return result


def tile_positions(length, side, overlap=0.5):
    if length <= side:
        return [0]
    stride = max(1, math.floor(side * (1 - overlap)))
    return sorted(set(list(range(0, length - side + 1, stride)) + [length - side]))


def blend_weights(side, device, floor=0.001):
    u = (torch.arange(side, device=device, dtype=torch.float32) + 0.5 - side / 2) / (
        side / 4
    )
    return torch.exp(-0.5 * (u[:, None] ** 2 + u[None, :] ** 2)).clamp_min(floor)
