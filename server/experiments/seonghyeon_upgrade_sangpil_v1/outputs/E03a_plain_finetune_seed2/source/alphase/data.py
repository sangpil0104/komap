from collections import Counter
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import cv2
from PIL import Image
from .contracts import rgb_to_ids, d4_forward
from .geometry import native_side, native_target, prepare_inputs
from .io import jsonl_read, local_path, sha256, ids_hash


def dataset_rows(cfg, split):
    rows = [
        r
        for r in jsonl_read(local_path(cfg["paths"]["dataset_manifest"]))
        if r["split"] == split
    ]
    if split in ["train", "valid"]:
        ids = cfg["training"][split + "_ids"]
        if ids is not None:
            found = {r["image_id"] for r in rows}
            if not set(ids) <= found or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid {split}_ids")
            rows = [r for r in rows if r["image_id"] in ids]
    return rows


def verify_rows(rows, split=None):
    if not rows:
        raise ValueError("Empty dataset selection")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    for r in rows:
        if split and r["split"] != split:
            raise ValueError(f'{split}-only data required: {r["image_id"]}')
        for kind in ["image", "mask"]:
            if r.get(kind + "_path"):
                p = local_path(r[kind + "_path"])
                if not p.is_file() or sha256(p) != r[kind + "_sha256"]:
                    raise ValueError(f"Stale {kind}: {p}")


def boundary_mask(target):
    valid = target != 255
    b = np.zeros_like(valid)
    d = (target[1:] != target[:-1]) & valid[1:] & valid[:-1]
    b[1:] |= d
    b[:-1] |= d
    d = (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    return b


def mixed_scores(target, window=41):
    valid = (target != 255).astype(np.float32)
    norm = cv2.boxFilter(
        valid, -1, (window, window), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    p = np.stack(
        [
            cv2.boxFilter(
                (target == c).astype(np.float32),
                -1,
                (window, window),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            / np.maximum(norm, 1)
            for c in range(4)
        ]
    )
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(0) / np.log(4)
    density = cv2.boxFilter(
        boundary_mask(target).astype(np.float32),
        -1,
        (window, window),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    ) / np.maximum(norm, 1)
    return np.minimum(p[1], p[3]) * entropy * density * valid


def hard_scores(target, prob):
    valid = target != 255
    safe = target.copy()
    safe[~valid] = 0
    score = -np.log(np.maximum(np.take_along_axis(prob, safe[None], axis=0)[0], 1e-6))
    score = np.minimum(score, np.percentile(score[valid], 95))
    score[(target == 1) & (prob.argmax(0) == 3)] *= 2
    score[~valid] = 0
    return score


def sample_seed(seed, step, micro, sample, rank=0):
    text = f"{seed}:{step}:{micro}:{rank}:{sample}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def legacy_augmentation(crop, seed):
    return A.Compose(
        [
            A.Affine(
                scale=(0.7, 1.4),
                rotate=(-180, 180),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(crop, crop),
            A.D4(),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
            A.RandomGamma(gamma_limit=(60, 160), p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.01, 0.05)),
                ],
                p=0.3,
            ),
        ],
        seed=seed,
    )


def native_photo(gray, seed, blur_noise=False):
    ops = [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
    ]
    if blur_noise:
        ops.append(
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.01, 0.05)),
                ],
                p=0.3,
            )
        )
    return A.Compose(ops, seed=seed)(image=gray)["image"]


class TrainSampler:
    """Every choice is indexed by update/micro/sample, independent of prefetch/RNG."""

    def __init__(self, cfg, rows=None):
        self.cfg = cfg
        self.rows = dataset_rows(cfg, "train") if rows is None else rows
        verify_rows(self.rows, "train")
        self.gray = [
            np.array(Image.open(r["image_path"]).convert("L")) for r in self.rows
        ]
        self.target = [
            rgb_to_ids(np.array(Image.open(r["mask_path"]))) for r in self.rows
        ]
        areas = np.array([a.size for a in self.gray], np.float64)
        self.weights = (
            areas / areas.sum()
            if cfg["sampling"]["image_selection"] == "area"
            else None
        )
        self.mixed = (
            [
                mixed_scores(y, cfg["sampling"]["local_window_native_px"])
                for y in self.target
            ]
            if cfg["sampling"]["mixed"] > 0
            else None
        )
        self.hard = {}
        self.hard_steps = {}
        self.mining_callback = None
        self.counts = Counter()
        self.hard_manifest_hash = None
        if cfg["sampling"]["hard_cache"] is not None:
            from .cache import load_probability

            path = local_path(cfg["sampling"]["hard_cache"])
            self.hard_manifest_hash = sha256(path)
            lookup = {r["image_id"]: i for i, r in enumerate(self.rows)}
            for row in jsonl_read(path):
                if row["source_split"] != "train" or row["image_id"] not in lookup:
                    raise ValueError("Hard cache contains non-training data")
                is_oof = "lineage" in row
                if is_oof:
                    from .oof import validate_lineage

                    validate_lineage(row["lineage"], self.rows)
                elif not cfg["sampling"]["allow_train_in_sample_mining"]:
                    raise ValueError("In-sample hard mining disabled")
                i = lookup[row["image_id"]]
                if row["image_sha256"] != self.rows[i]["image_sha256"]:
                    raise ValueError("Hard cache image mismatch")
                prob = load_probability(row, path.parent)
                self.hard[i] = hard_scores(self.target[i], prob)
            self.hard_policy = (
                "strict_oof"
                if all("lineage" in r for r in jsonl_read(path))
                else "in_sample_train_mining"
            )
        else:
            self.hard_policy = (
                "online_in_sample_train_mining_after_warmup"
                if cfg["sampling"]["allow_train_in_sample_mining"]
                else "no_cache_random_same_image_fallback"
            )

    def state_dict(self):
        return {
            "counts": dict(self.counts),
            "hard_manifest_hash": self.hard_manifest_hash,
            "hard_policy": self.hard_policy,
            "hard_steps": self.hard_steps,
            "online_hard": (
                {i: torch.from_numpy(score) for i, score in self.hard.items()}
                if self.hard_manifest_hash is None
                else {}
            ),
        }

    def load_state_dict(self, state):
        if state["hard_manifest_hash"] != self.hard_manifest_hash:
            raise ValueError("Hard cache changed across resume")
        self.counts = Counter(state["counts"])
        self.hard_steps = state.get("hard_steps", {})
        if self.hard_manifest_hash is None:
            self.hard = {i: s.numpy() for i, s in state.get("online_hard", {}).items()}

    def sample(self, step, micro, index, device):
        cfg = self.cfg
        g = cfg["geometry"]
        seed = sample_seed(cfg["run"]["seed"], step, micro, index)
        rng = np.random.default_rng(seed)
        i = int(rng.choice(len(self.rows), p=self.weights))
        row = self.rows[i]
        gray = self.gray[i]
        target = self.target[i]
        h, w = gray.shape
        p = g["detail_input_size"]
        if g["profile"] == "legacy_scaled_crop":
            scale = g["target_width"] / w
            s = int(p * 2.1 / scale)
            im = gray
            mask = target
            if h > s and w > s:
                y = int(rng.integers(0, h - s + 1))
                x = int(rng.integers(0, w - s + 1))
                im = im[y : y + s, x : x + s]
                mask = mask[y : y + s, x : x + s]
            if scale != 1:
                im = cv2.resize(
                    im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
                )
                mask = cv2.resize(
                    mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            out = legacy_augmentation(p, seed)(image=im, mask=mask)
            x = (out["image"].astype(np.float32) / 255.0 - 0.449) / 0.226
            return {
                "image_id": row["image_id"],
                "detail_rgb": torch.from_numpy(np.stack([x, x, x]))[None].to(device),
                "target_detail": torch.from_numpy(out["mask"].astype(np.int64)).to(
                    device
                ),
                "geometry": {"profile": "legacy_scaled_crop"},
                "target_native": None,
            }
        n = native_side(w, p, g["target_width"])
        mode = str(
            rng.choice(
                ["random", "mixed", "hard"],
                p=[cfg["sampling"][k] for k in ["random", "mixed", "hard"]],
            )
        )
        if (
            mode == "hard"
            and self.hard_manifest_hash is None
            and self.mining_callback is not None
            and cfg["sampling"]["allow_train_in_sample_mining"]
            and step >= cfg["sampling"]["hard_warmup_updates"]
            and (
                i not in self.hard_steps
                or step - self.hard_steps[i] >= cfg["sampling"]["hard_refresh_updates"]
            )
        ):
            # Callback gets only this TRAIN image. GT is used here solely to rank
            # sampling centers; it never enters model input or normalization.
            probability = self.mining_callback(gray)
            self.hard[i] = hard_scores(target, probability)
            self.hard_steps[i] = step
            self.counts["online_hard_refresh"] += 1
        score = (
            self.mixed[i]
            if mode == "mixed"
            else self.hard.get(i) if mode == "hard" else None
        )
        if mode != "random" and (score is None or score.sum() <= 0):
            self.counts[mode + "_fallback"] += 1
            mode = "random"
        self.counts[mode] += 1
        if mode == "random":
            x = int(rng.integers(0, max(w - n, 0) + 1))
            y = int(rng.integers(0, max(h - n, 0) + 1))
        else:
            loc = int(
                rng.choice(
                    score.size,
                    p=(score.astype(np.float64) / score.sum(dtype=np.float64)).ravel(),
                )
            )
            cy, cx = divmod(loc, w)
            x = int(np.clip(cx - n // 2, 0, max(w - n, 0)))
            y = int(np.clip(cy - n // 2, 0, max(h - n, 0)))
        d = (x, y, n, n)
        k = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        photo = native_photo(gray, seed, cfg["training"]["native_blur_noise"])
        image = torch.from_numpy(photo.copy()).to(device).float()[None, None] / 255
        sample = prepare_inputs(image, d, cfg, k, flip)
        native, valid = native_target(
            torch.from_numpy(target.astype(np.int64)).to(device), d
        )
        detail = F.interpolate(
            native[None, None].float(), size=(p, p), mode="nearest-exact"
        )[0, 0].long()
        sample.update(
            image_id=row["image_id"],
            target_detail=d4_forward(detail, k, flip),
            target_native=native,
            valid_native=valid,
            geometry={
                "profile": "native_roi_v1",
                "native_hw": [h, w],
                "detail_rect_xywh": list(d),
                "context_rect_xywh": sample.get("context_rects", torch.empty(0))
                .flatten()
                .tolist()
                or None,
                "d4_k": k,
                "d4_flip": flip,
                "original_width_reference": w,
            },
        )
        return sample

    def batch(self, step, micro, device):
        samples = [
            self.sample(step, micro, j, device)
            for j in range(self.cfg["training"]["batch_size"])
        ]
        result = {
            "image_ids": [s["image_id"] for s in samples],
            "target_detail": torch.stack([s["target_detail"] for s in samples]),
            "target_native": [s["target_native"] for s in samples],
            "geometry": [s["geometry"] for s in samples],
        }
        for key in [
            "detail_rgb",
            "texture_detail",
            "context_gray",
            "detail_rects",
            "context_rects",
        ]:
            if key in samples[0]:
                result[key] = torch.cat([s[key] for s in samples])
        result["d4"] = [s.get("d4", [(0, False)])[0] for s in samples]
        return result


def model_inputs(batch):
    return {
        k: batch[k]
        for k in [
            "detail_rgb",
            "texture_detail",
            "context_gray",
            "detail_rects",
            "context_rects",
            "d4",
        ]
        if k in batch
    }
