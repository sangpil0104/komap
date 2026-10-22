"""Exact CropDS, augmentation, worker seeding and Dice functions copied from source Model326.
Only configure() and imports are adapters; never import upstream train.py (it starts training).
Linux fork workers inherit this original module-level configuration.
"""
import random
import time
from types import SimpleNamespace
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from runtime import common
NUM_CLASSES = common.NUM_CLASSES
load_gray, load_mask = common.load_gray, common.load_mask
resize_gray, make_input = common.resize_gray, common.make_input
args = aug = None

def configure(config):
    global args, aug
    args = SimpleNamespace(**config)
    aug = A.Compose([
        A.Affine(scale=(0.7, 1.4), rotate=(-180, 180), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.PadIfNeeded(args.crop, args.crop, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(args.crop, args.crop),
        A.D4(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise(std_range=(0.01, 0.05))], p=0.3),
    ])


class CropDS(torch.utils.data.Dataset):
    def __init__(self, pairs, n):
        self.imgs = [load_gray(i) for i, _ in pairs]
        self.masks = [load_mask(m) for _, m in pairs]
        area = np.array([im.size for im in self.imgs], np.float64)
        self.w = np.full(len(area), 1 / len(area)) if args.sample == "uniform" else area / area.sum()
        # per-image resize factor: magnification normalization, or the fixed --scale
        self.scales = [args.target_width / im.shape[1] if args.target_width else args.scale
                       for im in self.imgs]
        self.image_stats = []
        self.image_cdfs = []
        if args.norm == "image_global" or args.input == "zcdf3":
            for im, scale in zip(self.imgs, self.scales):
                resized = resize_gray(im, scale)
                values = resized.astype(np.float32) / 255.0
                self.image_stats.append((float(values.mean()), float(values.std())))
                histogram = np.bincount(resized.reshape(-1), minlength=256).astype(np.float64)
                self.image_cdfs.append(
                    (np.cumsum(histogram) / max(float(histogram.sum()), 1.0)).astype(np.float32))
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        i = np.random.choice(len(self.imgs), p=self.w)
        im, m = self.imgs[i], self.masks[i]
        scale = self.scales[i]
        # pre-crop a generous window so Affine is cheap and has context for rotation
        s = int(args.crop * 2.1 / scale)  # window in original pixels
        H, W = im.shape
        if H > s and W > s:
            y, x = np.random.randint(0, H - s + 1), np.random.randint(0, W - s + 1)
            im, m = im[y:y + s, x:x + s], m[y:y + s, x:x + s]
        if scale != 1:
            im = resize_gray(im, scale)
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        r = aug(image=im, mask=m)
        has_full_stats = args.norm == "image_global" or args.input == "zcdf3"
        image_stats = self.image_stats[i] if has_full_stats else None
        cdf_lut = self.image_cdfs[i] if args.input == "zcdf3" else None
        x = torch.from_numpy(make_input(r["image"], args.norm, args.input,
                                        args.feat_stats, image_stats, cdf_lut))
        return x, torch.from_numpy(r["mask"]).long()


def worker_init(wid):
    seed = args.seed * 1000 + wid
    if not args.deterministic_workers:
        seed = (seed + int(time.time())) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def dice_loss(logits, y):
    prob = logits.float().softmax(1)
    oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (prob * oh).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()
