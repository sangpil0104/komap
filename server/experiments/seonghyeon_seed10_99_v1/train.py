"""Independent seeded training; architecture and inference come from frozen vendor code."""
import argparse
import math
import os
from pathlib import Path
import random
import time
import traceback

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from report import comparison, scores, seed_report
from runtime import Progress, read_json, write_json
from vendor.common import (PALETTE, build_model, list_split, load_gray, load_mask,
                           make_input, make_member, predict_probs, resize_gray)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def augmentation(crop, seed):
    return A.Compose([
        A.Affine(scale=(0.7, 1.4), rotate=(-180, 180), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(crop, crop), A.D4(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)),
                 A.GaussNoise(std_range=(0.01, 0.05))], p=0.3),
    ], seed=seed)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, cfg):
        self.cfg = cfg
        self.images = [load_gray(i) for i, _ in pairs]
        self.masks = [load_mask(m) for _, m in pairs]
        area = np.array([im.size for im in self.images], np.float64)
        self.weights = area / area.sum()
        self.scales = [cfg["target_width"] / im.shape[1] for im in self.images]
        self.aug = augmentation(cfg["crop"], cfg["seed"])

    def __len__(self):
        return self.cfg["iters"] * self.cfg["bs"]

    def __getitem__(self, _):
        i = np.random.choice(len(self.images), p=self.weights)
        im, mask, scale = self.images[i], self.masks[i], self.scales[i]
        s = int(self.cfg["crop"] * 2.1 / scale)
        h, w = im.shape
        if h > s and w > s:
            y, x = np.random.randint(0, h - s + 1), np.random.randint(0, w - s + 1)
            im, mask = im[y:y+s, x:x+s], mask[y:y+s, x:x+s]
        if scale != 1:
            im = resize_gray(im, scale)
            mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = self.aug(image=im, mask=mask)
        return (torch.from_numpy(make_input(result["image"], "imagenet")),
                torch.from_numpy(result["mask"]).long())


def worker_init(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)
    torch.utils.data.get_worker_info().dataset.aug.set_random_seed(seed)


def dice_loss(logits, target):
    prob = logits.float().softmax(1)
    onehot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


def evaluate(model, cfg, pairs, progress, final=False):
    model.eval()
    member = make_member(model, "imagenet", target_width=cfg["target_width"])
    phase = "valid_d4" if final else "valid_plain"
    progress.update(state=phase, eval_image=0)
    progress.log(f"[{phase}] start 0/{len(pairs)}")
    rows, started = [], time.monotonic()
    if final:
        panel_dir = Path(cfg["result"]) / "valid_mask_comparison" / f"seed_{cfg['seed']:02d}"
        pred_dir = Path(cfg["result"]) / "predictions" / f"seed_{cfg['seed']:02d}"
        panel_dir.mkdir(parents=True, exist_ok=False)
        pred_dir.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for index, (ip, mp) in enumerate(pairs, 1):
            gray, target = load_gray(ip), load_mask(mp)
            pred = predict_probs([member], gray, tta=final).argmax(0).cpu().numpy()
            row = dict(image=Path(ip).name, **scores(pred, target))
            rows.append(row)
            if final:
                filename = Path(ip).stem + ".png"
                Image.fromarray(PALETTE[pred]).save(pred_dir / filename)
                with Image.open(ip) as original:
                    comparison(panel_dir / filename, np.array(original.convert("RGB")),
                               pred, target, row["image_miou"])
            rate_key = "d4_seconds_per_image" if final else "plain_seconds_per_image"
            progress.update(eval_image=index, **{rate_key: (time.monotonic() - started) / index})
            progress.log(f"[{phase}] {index}/{len(pairs)} {Path(ip).name} "
                         f"image_mIoU={row['image_miou'] * 100:.4f}%")
    if not final:
        progress.update(plain_evaluations_done=progress.state["plain_evaluations_done"] + 1)
    progress.log(f"[{phase}] mean_image_mIoU={np.mean([r['image_miou'] for r in rows])*100:.4f}% "
                 f"exclude_reference={np.mean([r['image_miou_exclude'] for r in rows])*100:.4f}%")
    model.train()
    return rows


def train(cfg, progress):
    run = Path(cfg["run"])
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    seed_all(cfg["seed"])
    if torch.cuda.device_count() != 1:
        raise RuntimeError("A seed worker must see exactly its one assigned GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required")
    pairs = list_split(cfg["data"], "train")
    valid_pairs = list_split(cfg["data"], "valid")
    if len(pairs) != 70 or len(valid_pairs) != 20:
        raise ValueError("Expected train 70 / valid 20")
    # args.json remains compatible with the original common.load_member/predict.py.
    write_json(run / "args.json", dict(cfg, arch="unet", encoder="efficientnet-b4", norm="imagenet",
                                      scale=1.0, input="gray3", feat_stats=None, sample="area",
                                      train_splits="train", lr=3e-4, wd=1e-2, dice_w=1.0,
                                      accum=1, ema=0.0, cldice_w=0.0))
    dataset = CropDataset(pairs, cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["bs"], num_workers=cfg["workers"], pin_memory=True,
        worker_init_fn=worker_init, persistent_workers=cfg["workers"] > 0, drop_last=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        **({"prefetch_factor": 1} if cfg["workers"] else {}))
    # Decoder initialization follows this run's seed; only ImageNet encoder is shared.
    model = build_model("unet", "efficientnet-b4", pretrained=False)
    model.encoder.load_state_dict(torch.load(cfg["encoder_weights"], map_location="cpu", weights_only=True))
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 300) *
        0.5 * (1 + math.cos(math.pi * min(it, cfg["iters"]) / cfg["iters"])))
    loss_average = 0.0
    train_seconds = 0.0
    progress.update(state="training")
    progress.log(f"[train] start seed={cfg['seed']} iterations={cfg['iters']} GPU={torch.cuda.get_device_name(0)}")
    mark = time.monotonic()
    for iteration, (x, y) in enumerate(loader, 1):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y) + dice_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration}: {value}")
        loss_average = 0.98 * loss_average + 0.02 * value if iteration > 1 else value
        train_seconds += time.monotonic() - mark
        del x, y, logits, loss
        if iteration == 1 or iteration % 20 == 0 or iteration == cfg["iters"]:
            progress.update(state="training", iteration=iteration, train_seconds=train_seconds,
                            seconds_per_iteration=train_seconds / iteration, loss=loss_average)
        if iteration == 1 or iteration % 100 == 0 or iteration == cfg["iters"]:
            progress.log(f"[train] {iteration}/{cfg['iters']} ({100*iteration/cfg['iters']:.2f}%) "
                         f"loss={loss_average:.4f} lr={sched.get_last_lr()[0]:.2e} "
                         f"s/it={train_seconds/iteration:.3f} "
                         f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
        if iteration % cfg["eval_every"] == 0 or iteration == cfg["iters"]:
            evaluate(model, cfg, valid_pairs, progress)
            progress.update(state="training")
        mark = time.monotonic()
    # Release CPU prefetch workers before the final evaluation.
    del loader, dataset
    progress.update(state="saving")
    tmp = run / "last.pt.tmp"
    torch.save(model.state_dict(), tmp)
    tmp.replace(run / "last.pt")
    rows = evaluate(model, cfg, valid_pairs, progress, final=True)
    progress.update(state="reporting")
    result = seed_report(cfg["result"], cfg["seed"], rows, run)
    progress.update(state="completed", image_miou=result["image_miou"])
    progress.log(f"[DONE] seed={cfg['seed']} image_mIoU={result['image_miou']*100:.4f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    cfg = read_json(parser.parse_args().job)
    progress = Progress(cfg["run"], cfg)
    try:
        train(cfg, progress)
    except BaseException as exc:
        progress.update(state="failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
