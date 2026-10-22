import numpy as np
import torch

from alphase.data.dataset import TrainDataset, load_image, load_target, build_inputs, pad_array
from alphase.data.geometry import crop_array
from pathlib import Path
from scipy.ndimage import gaussian_filter

def load_hard(folder, image_id):
    with np.load(Path(folder) / f"{image_id}.npz", allow_pickle=False) as data:
        return data["hard"].copy()


def d4_array(array, k):
    array = np.rot90(array, k % 4)
    return array[:, ::-1] if k // 4 else array


class HundredDataset(TrainDataset):
    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        aug = self.cfg["augmentation"]
        if "sample_class" in aug:
            fractions = np.array([float(r[f"phase{aug['sample_class']}_fraction"]) for r in self.rows])
            prob = np.sqrt(fractions + .01)
            row = self.rows[int(rng.choice(len(self.rows), p=prob/prob.sum()))]
        else:
            row = self.rows[int(rng.integers(len(self.rows)))]
        g = load_image(row["image_path"]).astype(np.float32) / 255
        target = load_target(row["mask_path"]).copy()
        hard = None
        if self.cfg["hundred"]["hard"] and self.stage["mode"] != "full_frame":
            hard = load_hard(self.cfg["hundred"]["mining_folder"], row["image_id"])
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
        if rng.random() < aug.get("extra_probability", 0.):
            if aug.get("noise_std"): g = g + rng.normal(0, aug["noise_std"], g.shape)
            if aug.get("blur_sigma"): g = gaussian_filter(g, aug["blur_sigma"])
            if aug.get("sharpen"): g = g + aug["sharpen"]*(g-gaussian_filter(g, 1.))
            if aug.get("local_contrast"): g = g + aug["local_contrast"]*(g-gaussian_filter(g, 8.))
            g = np.clip(g, 0, 1)
        detail, dino = build_inputs(np.ascontiguousarray(g, dtype=np.float32), self.raw_only, self.hybrid)
        if "drop_channel" in aug and rng.random() < aug.get("extra_probability", .25):
            detail[aug["drop_channel"]].zero_()
        h, w = g.shape
        if aug.get("erase_fraction") and rng.random() < aug.get("extra_probability", .25):
            side = np.sqrt(aug["erase_fraction"])
            eh, ew = max(1,int(h*side)), max(1,int(w*side))
            y, x = int(rng.integers(h-eh+1)), int(rng.integers(w-ew+1))
            detail[:,y:y+eh,x:x+ew] = detail[:,:h,:w].mean((1,2),keepdim=True)
            target = target.copy()
            target[y:y+eh,x:x+ew] = 255
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
