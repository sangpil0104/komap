import numpy as np
import torch

from alphase.data.dataset import TrainDataset, load_image, load_target, build_inputs, pad_array
from alphase.data.geometry import crop_array
from refinelab.mining import load_hard


def d4_array(array, k):
    array = np.rot90(array, k % 4)
    return array[:, ::-1] if k // 4 else array


class RefineDataset(TrainDataset):
    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        g = load_image(row["image_path"]).astype(np.float32) / 255
        target = load_target(row["mask_path"]).copy()
        hard = None
        if self.cfg["refinement"]["hard"] and self.stage["mode"] != "full_frame":
            hard = load_hard(self.cfg["refinement"]["mining_folder"], row["image_id"])
            if hard.shape != target.shape:
                raise ValueError("Hard-example map must match the native image")
        aug = self.cfg["augmentation"]
        if aug["d4"]:
            k = int(rng.integers(8))
            g, target = d4_array(g, k), d4_array(target, k)
            if hard is not None:
                hard = d4_array(hard, k)
        if rng.random() < self.stage.get("photometric_probability", aug["photometric_probability"]):
            g = np.clip((g-.5) * rng.uniform(*aug["contrast_gain"]) + .5 + rng.uniform(*aug["brightness_offset"]), 0, 1)
            g = np.power(g, rng.uniform(*aug["gamma"]))
        detail, dino = build_inputs(np.ascontiguousarray(g, dtype=np.float32), self.raw_only, self.hybrid)
        h, w = g.shape
        if self.stage["mode"] == "full_frame":
            mask = pad_array(target, 255).astype(np.int64)
            box = (0, 0, detail.shape[-1], detail.shape[-2])
        else:
            core = int(rng.choice(self.stage["core_sizes"]))
            halo = self.cfg["sampling"]["halo"]
            if core % 32 or (core + 2*halo) % 32:
                raise ValueError("Core/halo input must be multiples of 32")
            mode, candidates = rng.random(), None
            mix = self.cfg["sampling"]["mix"]
            if mode < mix["eutectic_center"]:
                candidates = np.flatnonzero(target == 2)
            elif mode < mix["eutectic_center"] + mix["error_or_boundary"]:
                if hard is not None:
                    candidates = np.flatnonzero(hard)
                if candidates is None or not len(candidates):
                    boundary = np.zeros(target.shape, bool)
                    boundary[1:] |= target[1:] != target[:-1]
                    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
                    candidates = np.flatnonzero(boundary)
            if candidates is not None and len(candidates):
                cy, cx = np.unravel_index(rng.choice(candidates), target.shape)
            else:
                cy, cx = rng.integers(h), rng.integers(w)
            x, y = int(cx)-core//2, int(cy)-core//2
            box = (x-halo, y-halo, x+core+halo, y+core+halo)
            detail = torch.from_numpy(crop_array(detail.numpy()[..., :h, :w], box))
            mask = np.full((core+2*halo, core+2*halo), 255, dtype=np.int64)
            mask[halo:halo+core, halo:halo+core] = crop_array(target, (x, y, x+core, y+core), 255)
        if not (mask != 255).any():
            raise ValueError("Empty target")
        return dict(detail=detail, dino=dino, target=torch.from_numpy(mask),
                    box=torch.tensor(box, dtype=torch.float32), image_id=row["image_id"])
