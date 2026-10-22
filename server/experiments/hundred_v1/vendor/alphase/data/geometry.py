import numpy as np
import torch
import torch.nn.functional as F


def d4(x, index):
    result = torch.rot90(x, index % 4, (-2, -1))
    return result.flip(-1) if index // 4 else result


def undo_d4(x, index):
    if index // 4:
        x = x.flip(-1)
    return torch.rot90(x, -(index % 4), (-2, -1))


def pad_image(x, multiple=32):
    h, w = x.shape[-2:]
    ph, pw = -h % multiple, -w % multiple
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode)


def crop_array(array, box, fill=None):
    """CHW or HW crop in image-edge coordinates, including outside-image halo."""
    x0, y0, x1, y1 = map(int, box)
    h, w = array.shape[-2:]
    left, top, right, bottom = max(-x0, 0), max(-y0, 0), max(x1-w, 0), max(y1-h, 0)
    padding = [(0, 0)] * (array.ndim-2) + [(top, bottom), (left, right)]
    if fill is not None:
        padded = np.pad(array, padding, mode="constant", constant_values=fill)
    else:
        padded = np.pad(array, padding, mode="reflect" if min(h, w) > 1 else "edge")
    return padded[..., y0+top:y1+top, x0+left:x1+left].copy()


def tile_starts(length, core, stride):
    if stride < 1 or stride > core:
        raise ValueError("Tile stride must be within [1, core]")
    last = max(length-core, 0)
    return sorted(set(list(range(0, last+1, stride)) + [last]))
