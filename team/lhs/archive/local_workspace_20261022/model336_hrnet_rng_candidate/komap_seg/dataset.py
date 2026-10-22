"""Patch-based training dataset and full-resolution evaluation dataset."""

import random
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from komap_seg.palette import rgb_to_class

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Al3Ni and Eutectic Si are the rarest phases. Half of the training patches
# are centered near a pixel of one of them (positive sampling), chosen with
# equal probability. This barely changes the class frequency the model sees
# (patches are large), it just guarantees the rare phases show up often.
MINORITY_CLASSES = (1, 3)
MINORITY_CLASS_WEIGHTS = {1: 1.0, 3: 1.0}

# Cap on stored coordinates per (image, minority class) to bound memory.
MAX_MINORITY_COORDS = 4000


def find_pairs(images_dir: Path, masks_dir: Optional[Path]) -> list[dict]:
    """`*_image.png` files and their `*_mask.png` partners (same stem)."""
    images_dir = Path(images_dir)
    pairs = []
    for img_path in sorted(images_dir.glob("*_image.png")):
        stem = img_path.name[: -len("_image.png")]
        mask_path = None
        if masks_dir is not None:
            mask_path = Path(masks_dir) / f"{stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"mask not found for {img_path}: {mask_path}")
        pairs.append({"stem": stem, "image": img_path, "mask": mask_path})
    if not pairs:
        raise FileNotFoundError(f"no *_image.png files found under {images_dir}")
    return pairs


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_class_mask(path: Path) -> np.ndarray:
    return rgb_to_class(np.array(Image.open(path).convert("RGB")))


def build_train_transform() -> A.Compose:
    """Augmentations applied to an already-cropped (and upscaled) patch."""
    reflect = cv2.BORDER_REFLECT_101
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.8, 1.2), border_mode=reflect, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=30, sigma=6, border_mode=reflect, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=reflect, p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: reseed python/numpy RNG per worker (fork
    otherwise duplicates the parent's RNG state into every worker)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def derive_item_seed(run_seed: int, epoch: int, index: int, stream_id: int) -> int:
    """Stable RNG seed independent of DataLoader worker assignment/order."""
    seed_words = [int(run_seed) % (2**64), int(epoch), int(index), int(stream_id)]
    state = np.random.SeedSequence(seed_words).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def seed_transform_tree(transform: A.Compose, seed: int) -> None:
    """Seed Compose and each nested transform with a distinct deterministic seed.

    Albumentations 2.0.8 propagates Compose.set_random_seed(seed) to every child
    using that same seed. Re-seeding each node separately avoids correlating
    equal-probability gates such as HorizontalFlip and VerticalFlip.
    """
    set_seed = getattr(transform, "set_random_seed", None)
    if not callable(set_seed):
        raise RuntimeError(
            "per-item RNG mode requires Albumentations Compose.set_random_seed"
        )
    set_seed(int(seed))

    nodes = list(getattr(transform, "transforms", ()))
    node_index = 0
    while nodes:
        node = nodes.pop(0)
        node_set_seed = getattr(node, "set_random_seed", None)
        if not callable(node_set_seed):
            raise RuntimeError(
                f"per-item RNG mode cannot seed transform node {type(node).__name__}"
            )
        node_seed = derive_item_seed(seed, epoch=0, index=node_index, stream_id=2)
        node_set_seed(node_seed)
        node_index += 1
        # A BaseCompose reseeds its descendants too; those nodes are later
        # visited here and assigned their own independent deterministic seeds.
        nodes.extend(getattr(node, "transforms", ()))


class KomapPatchDataset(Dataset):
    """Random-crop patch dataset for training.

    All (image, mask) pairs are cached in memory. Each item crops a
    `patch_size / input_scale` native region (uniformly at random, or
    near a rare-phase pixel with probability `positive_sample_prob`),
    upscales it to `patch_size` when `input_scale != 1`, then augments.
    Length is virtual: `len(images) * samples_per_image`.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        patch_size: int = 512,
        samples_per_image: int = 12,
        positive_sample_prob: float = 0.5,
        input_scale: float = 2.0,
        transform: Optional[A.Compose] = None,
        seed: Optional[int] = None,
        rng_mode: str = "legacy",
    ):
        self.patch_size = patch_size
        self.input_scale = input_scale
        self.crop_size = round(patch_size / input_scale)
        self.samples_per_image = samples_per_image
        self.positive_sample_prob = positive_sample_prob
        self.transform = transform if transform is not None else build_train_transform()
        self.rng = random.Random(seed)
        if rng_mode not in {"legacy", "per_item"}:
            raise ValueError("rng_mode must be 'legacy' or 'per_item'")
        self.rng_mode = rng_mode
        self.run_seed = 0 if seed is None else int(seed)
        self.epoch = 0

        pairs = find_pairs(images_dir, masks_dir)
        self.stems = [p["stem"] for p in pairs]
        self.images: list[np.ndarray] = []
        self.class_masks: list[np.ndarray] = []
        self.minority_coords: list[dict[int, np.ndarray]] = []

        for p in pairs:
            image = load_image_rgb(p["image"])
            class_mask = load_class_mask(p["mask"])

            # Reflect-pad images smaller than the crop so every image trains.
            pad_h = max(self.crop_size - image.shape[0], 0)
            pad_w = max(self.crop_size - image.shape[1], 0)
            if pad_h or pad_w:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                class_mask = np.pad(class_mask, ((0, pad_h), (0, pad_w)), mode="reflect")

            self.images.append(image)
            self.class_masks.append(class_mask)

            coords: dict[int, np.ndarray] = {}
            for cls in MINORITY_CLASSES:
                ys, xs = np.where(class_mask == cls)
                if len(ys) == 0:
                    continue
                pts = np.stack([ys, xs], axis=1)
                if len(pts) > MAX_MINORITY_COORDS:
                    idx = self.rng.sample(range(len(pts)), MAX_MINORITY_COORDS)
                    pts = pts[idx]
                coords[cls] = pts
            self.minority_coords.append(coords)

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_image

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_top_left(
        self,
        img_idx: int,
        h: int,
        w: int,
        rng: Optional[random.Random] = None,
    ) -> tuple[int, int]:
        ps = self.crop_size
        rng = self.rng if rng is None else rng
        coords = self.minority_coords[img_idx]
        if coords and rng.random() < self.positive_sample_prob:
            keys = list(coords.keys())
            weights = [MINORITY_CLASS_WEIGHTS.get(k, 1.0) for k in keys]
            cls = rng.choices(keys, weights=weights, k=1)[0]
            pts = coords[cls]
            cy, cx = pts[rng.randrange(len(pts))]
            jitter = ps // 4
            top = cy - ps // 2 + rng.randint(-jitter, jitter)
            left = cx - ps // 2 + rng.randint(-jitter, jitter)
            top = int(min(max(top, 0), h - ps))
            left = int(min(max(left, 0), w - ps))
            return top, left

        top = rng.randint(0, h - ps)
        left = rng.randint(0, w - ps)
        return top, left

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rng_mode == "legacy":
            return self._get_item(index)

        crop_seed = derive_item_seed(self.run_seed, self.epoch, index, stream_id=0)
        transform_seed = derive_item_seed(self.run_seed, self.epoch, index, stream_id=1)
        return self._get_item(index, random.Random(crop_seed), transform_seed)

    def _get_item(
        self,
        index: int,
        crop_rng: Optional[random.Random] = None,
        transform_seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx = index % len(self.images)
        image = self.images[img_idx]
        class_mask = self.class_masks[img_idx]
        h, w = class_mask.shape

        top, left = self._sample_top_left(img_idx, h, w, rng=crop_rng)
        cs = self.crop_size
        img_patch = image[top : top + cs, left : left + cs]
        mask_patch = class_mask[top : top + cs, left : left + cs]

        if self.input_scale != 1.0:
            ps = self.patch_size
            img_patch = cv2.resize(img_patch, (ps, ps), interpolation=cv2.INTER_CUBIC)
            mask_patch = cv2.resize(mask_patch, (ps, ps), interpolation=cv2.INTER_NEAREST)

        if transform_seed is not None:
            seed_transform_tree(self.transform, transform_seed)
        out = self.transform(image=img_patch, mask=mask_patch)
        return out["image"].float(), out["mask"].long()


class KomapFullDataset(Dataset):
    """Full-resolution images (and masks, if given) for evaluation/prediction.

    Image sizes vary, so iterate it directly (or use batch_size=1).
    """

    def __init__(self, images_dir: Path, masks_dir: Optional[Path] = None, transform: Optional[A.Compose] = None):
        self.pairs = find_pairs(images_dir, masks_dir)
        self.transform = transform if transform is not None else build_eval_transform()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        p = self.pairs[index]
        image = load_image_rgb(p["image"])
        if p["mask"] is not None:
            out = self.transform(image=image, mask=load_class_mask(p["mask"]))
            mask_t = out["mask"].long()
        else:
            out = self.transform(image=image)
            mask_t = None
        return {
            "image": out["image"].float(),
            "mask": mask_t,
            "stem": p["stem"],
            "height": image.shape[0],
            "width": image.shape[1],
        }
