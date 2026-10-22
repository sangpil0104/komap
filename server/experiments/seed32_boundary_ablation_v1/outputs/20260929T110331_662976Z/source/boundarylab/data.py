"""Train-only, index-addressed sampling. Paired arms see matching random choices."""
import hashlib
import math
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io import sha256
from .metrics import boundary_band, mask_ids


def list_rows(root, split):
    root = Path(root)
    if split not in ('train', 'valid'):
        raise ValueError('Only official train/valid data are used')
    rows = []
    for image in sorted((root / split / 'images').glob('*_image.png')):
        image_id = image.name.removesuffix('_image.png')
        mask = root / split / 'masks' / (image_id + '_mask.png')
        if not mask.is_file():
            raise FileNotFoundError(mask)
        rows.append(dict(image_id=image_id, split=split, image=str(image), mask=str(mask)))
    if not rows:
        raise ValueError(f'No {split} data in {root}')
    return rows


def audit_data(root):
    rows = list_rows(root, 'train') + list_rows(root, 'valid')
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'valid')}
    if counts != dict(train=70, valid=20):
        raise ValueError(f'Expected train70/valid20: {counts}')
    if len({r['image_id'] for r in rows}) != 90:
        raise ValueError('Split overlap or duplicate ID')
    for r in rows:
        with Image.open(r['image']) as im:
            shape = (im.height, im.width)
        with Image.open(r['mask']) as mask:
            labels = mask_ids(np.array(mask))
        if labels.shape != shape:
            raise ValueError(f'Image-mask mismatch {r["image_id"]}')
        r.update(shape=list(shape), image_sha256=sha256(r['image']), mask_sha256=sha256(r['mask']))
    return rows


def indexed_seed(seed, index, stream):
    return int.from_bytes(hashlib.sha256(f'{seed}:{index}:{stream}'.encode()).digest()[:4], 'little')


def legacy_augmentation(crop, seed, gaussian=True):
    # Same operator ordering/parameters as the archived seed-32 training script.
    blur = A.GaussianBlur(blur_limit=(3, 5), p=0.5) if gaussian else A.NoOp(p=0.5)
    return A.Compose([
        A.Affine(scale=(0.7, 1.4), rotate=(-180, 180), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(crop, crop), A.D4(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
        A.OneOf([blur, A.GaussNoise(std_range=(0.01, 0.05), p=0.5)], p=0.3),
    ], seed=seed)


def native_roi(gray, target, center, side, angle=0.0, zoom=1.0):
    """Identical pixel-center affine for image/mask, expressed in native pixels."""
    cx, cy = center
    theta = math.radians(angle)
    c, s = math.cos(theta) / zoom, math.sin(theta) / zoom
    mid = (side - 1) / 2
    # WARP_INVERSE_MAP maps destination centers to source centers.
    matrix = np.array([[c, -s, cx-c*mid+s*mid], [s, c, cy-s*mid-c*mid]], np.float32)
    image = cv2.warpAffine(gray, matrix, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(target, matrix, (side, side), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return image, mask


def normalize(gray):
    return (gray.astype(np.float32) / 255.0 - 0.449) / 0.226


class TrainDataset(Dataset):
    def __init__(self, rows, cfg):
        if not rows or any(r['split'] != 'train' for r in rows):
            raise ValueError('Sampling must use ONLY train rows')
        self.rows, self.cfg = rows, cfg
        self.images = [np.array(Image.open(r['image']).convert('L')) for r in rows]
        self.masks = [mask_ids(np.array(Image.open(r['mask']))) for r in rows]
        area = np.array([x.size for x in self.images], np.float64)
        self.image_weights = area / area.sum()
        self.boundaries = [np.flatnonzero(boundary_band(y, 0)) for y in self.masks] if cfg['boundary_sampling'] else None

    def __len__(self):
        return self.cfg['steps'] * self.cfg['global_batch']

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        cfg, p = self.cfg, self.cfg['crop']
        seed = indexed_seed(cfg['seed'], int(index), 'sample')
        rng = np.random.default_rng(seed)
        i = int(rng.choice(len(self.rows), p=self.image_weights))
        gray, target = self.images[i], self.masks[i]
        h, w = gray.shape
        if cfg['geometry'] == 'legacy_scaled_crop':
            scale = cfg['target_width'] / w
            side = int(p * 2.1 / scale)
            if h > side and w > side:
                y, x = int(rng.integers(h-side+1)), int(rng.integers(w-side+1))
                gray, target = gray[y:y+side, x:x+side], target[y:y+side, x:x+side]
            if scale != 1:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                target = cv2.resize(target, gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
            out = legacy_augmentation(p, seed, cfg['gaussian_blur'])(image=gray, mask=target)
            detail, label = out['image'], out['mask']
            native_gray, native_label = None, None
            band = np.zeros((p, p), np.float32)
            geometry = dict(profile=cfg['geometry'])
        elif cfg['geometry'] == 'aligned_native_roi_v1':
            side = max(8, math.floor(p * w / cfg['target_width'] + 0.5))
            # Separate streams ensure changing crop selection does not change photometry.
            loc = np.random.default_rng(indexed_seed(cfg['seed'], int(index), 'location'))
            x = int(loc.integers(max(w-side, 0)+1))
            y = int(loc.integers(max(h-side, 0)+1))
            sampled_boundary = False
            if loc.random() < cfg['boundary_sampling'] and self.boundaries[i].size:
                pixel = int(loc.choice(self.boundaries[i]))
                cy, cx = divmod(pixel, w)
                x = int(np.clip(cx-side//2, 0, max(w-side, 0)))
                y = int(np.clip(cy-side//2, 0, max(h-side, 0)))
                sampled_boundary = True
            geom = np.random.default_rng(indexed_seed(cfg['seed'], int(index), 'geometry'))
            affine = geom.random() < 0.7
            angle = float(geom.uniform(-180, 180)) if affine else 0.0
            zoom = float(geom.uniform(0.7, 1.4)) if affine else 1.0
            native, label_native = native_roi(gray, target, (x+(side-1)/2, y+(side-1)/2), side, angle, zoom)
            k, flip = int(geom.integers(4)), bool(geom.integers(2))
            native, label_native = np.rot90(native, k), np.rot90(label_native, k)
            if flip:
                native, label_native = np.fliplr(native), np.fliplr(label_native)
            photo = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
                A.RandomGamma(gamma_limit=(60, 160), p=0.5),
            ], seed=seed)(image=np.ascontiguousarray(native))['image']
            detail = cv2.resize(photo, (p, p), interpolation=cv2.INTER_LINEAR)
            blur = A.GaussianBlur(blur_limit=(3,5), p=0.5) if cfg['gaussian_blur'] else A.NoOp(p=0.5)
            detail = A.Compose([A.OneOf([blur, A.GaussNoise(std_range=(0.01,0.05), p=0.5)], p=0.3)], seed=seed)(image=detail)['image']
            label_native = np.ascontiguousarray(label_native)
            label = cv2.resize(label_native, (p, p), interpolation=cv2.INTER_NEAREST_EXACT)
            band = cv2.resize(boundary_band(label_native, cfg['boundary_radius']).astype(np.float32),
                              (p, p), interpolation=cv2.INTER_NEAREST_EXACT)
            native_gray = torch.from_numpy(normalize(photo).copy())[None]
            native_label = torch.from_numpy(label_native.astype(np.int64))
            geometry = dict(profile=cfg['geometry'], rect=[x,y,side,side], angle=angle, zoom=zoom,
                            d4=[k,flip], sampled_boundary=sampled_boundary)
        else:
            raise ValueError(cfg['geometry'])
        if not (label < 4).any():
            raise ValueError('All-ignore training crop')
        x = normalize(detail)
        return dict(detail=torch.from_numpy(np.stack([x,x,x])), target=torch.from_numpy(label.astype(np.int64)),
                    native_gray=native_gray, native_target=native_label, boundary=torch.from_numpy(band),
                    image_id=self.rows[i]['image_id'], geometry=geometry, sample_index=int(index))


def one_sample_collate(samples):
    if len(samples) != 1:
        raise ValueError('Protocol uses one image per GPU and eight GPUs')
    sample = samples[0]
    return {k: v[None] if torch.is_tensor(v) else v for k, v in sample.items()}
