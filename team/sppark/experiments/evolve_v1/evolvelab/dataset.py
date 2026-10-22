"""Technique 2: a true 4-way crop-center mix (uniform / complex_region / boundary_rich /
hard_example) replacing hundred_v1's 3-way nested mix. `complex_region` centers crops on eutectic
Si (class 2, the "leopard-pattern" complex-texture class); `boundary_rich` centers crops on
class-transition pixels; `hard_example` centers crops on a per-parent mining map (evolvelab/mining.py),
falling back to boundary_rich when no map is available for an image."""
from pathlib import Path

import numpy as np
import torch

from alphase.data.dataset import TrainDataset, load_image, load_target, build_inputs, pad_array
from alphase.data.geometry import crop_array

CATEGORIES = ('uniform', 'complex_region', 'boundary_rich', 'hard_example')


def load_hard(folder, image_id):
    path = Path(folder) / f'{image_id}.npz'
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return data['hard'].copy()


def d4_array(array, k):
    array = np.rot90(array, k % 4)
    return array[:, ::-1] if k // 4 else array


def boundary_mask(target):
    boundary = np.zeros(target.shape, bool)
    boundary[1:] |= target[1:] != target[:-1]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    return boundary


def pick_category(rng, mix):
    weights = np.array([mix.get(c, 0.) for c in CATEGORIES], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        raise ValueError('sampling.mix must have positive total weight')
    return CATEGORIES[int(rng.choice(len(CATEGORIES), p=weights / total))]


class EvolveDataset(TrainDataset):
    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        g = load_image(row['image_path']).astype(np.float32) / 255
        target = load_target(row['mask_path']).copy()
        hard = None
        mining_folder = self.cfg['hundred'].get('mining_folder')
        if mining_folder and self.stage['mode'] != 'full_frame':
            hard = load_hard(mining_folder, row['image_id'])
        aug = self.cfg['augmentation']
        if aug['d4']:
            k = int(rng.integers(8))
            g, target = d4_array(g, k), d4_array(target, k)
            if hard is not None:
                hard = d4_array(hard, k)
        if rng.random() < self.stage.get('photometric_probability', aug['photometric_probability']):
            g = np.clip((g - .5) * rng.uniform(*aug['contrast_gain']) + .5 + rng.uniform(*aug['brightness_offset']), 0, 1)
            g = np.power(g, rng.uniform(*aug['gamma']))
        detail, dino = build_inputs(np.ascontiguousarray(g, dtype=np.float32), self.raw_only, self.hybrid)
        h, w = g.shape
        if self.stage['mode'] == 'full_frame':
            mask = pad_array(target, 255).astype(np.int64)
            box = (0, 0, detail.shape[-1], detail.shape[-2])
        else:
            core = int(rng.choice(self.stage['core_sizes']))
            halo = self.cfg['sampling']['halo']
            if core % 32 or (core + 2 * halo) % 32:
                raise ValueError('Core/halo input must be multiples of 32')
            category = pick_category(rng, self.cfg['sampling']['mix'])
            candidates = None
            if category == 'complex_region':
                candidates = np.flatnonzero(target == 2)
            elif category == 'boundary_rich':
                candidates = np.flatnonzero(boundary_mask(target))
            elif category == 'hard_example':
                candidates = np.flatnonzero(hard) if hard is not None else None
                if candidates is None or not len(candidates):
                    candidates = np.flatnonzero(boundary_mask(target))
            if candidates is not None and len(candidates):
                cy, cx = np.unravel_index(rng.choice(candidates), target.shape)
            else:
                cy, cx = rng.integers(h), rng.integers(w)
            x, y = int(cx) - core // 2, int(cy) - core // 2
            box = (x - halo, y - halo, x + core + halo, y + core + halo)
            detail = torch.from_numpy(crop_array(detail.numpy()[..., :h, :w], box))
            mask = np.full((core + 2 * halo, core + 2 * halo), 255, dtype=np.int64)
            mask[halo:halo + core, halo:halo + core] = crop_array(target, (x, y, x + core, y + core), 255)
        if not (mask != 255).any():
            raise ValueError('Empty target')
        return dict(detail=detail, dino=dino, target=torch.from_numpy(mask),
                    box=torch.tensor(box, dtype=torch.float32), image_id=row['image_id'])
