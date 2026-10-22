"""Train a phase-segmentation model on random crops at native resolution.

No checkpoint selection on valid: the last checkpoint is saved and valid is only reported.
"""
import argparse
import json
import math
import os
import random
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (NUM_CLASSES, build_model, image_miou, list_split, load_gray, load_mask,
                    make_input, make_member, morph_features, predict_probs, resize_gray)
from imagewise_lovasz import imagewise_lovasz_loss

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.expanduser("~/komap/data"))
p.add_argument("--out", required=True)
p.add_argument("--train_splits", default="train")
p.add_argument("--arch", default="unet")
p.add_argument("--encoder", default="tu-convnext_small")
p.add_argument("--norm", default="imagenet", choices=["imagenet", "image", "image_global"])
p.add_argument("--crop", type=int, default=512)
p.add_argument("--scale", type=float, default=1.0, help="train/infer on the image resized by this factor")
p.add_argument("--target_width", type=int, default=0,
               help="per-image magnification normalization: resize each image to this width (overrides --scale)")
p.add_argument("--sample", default="area", choices=["area", "uniform"],
               help="crop sampling: proportional to image area, or one image at a time (matches the per-image metric)")
p.add_argument("--bs", type=int, default=8)
p.add_argument("--iters", type=int, default=8000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--dice_w", type=float, default=1.0)
p.add_argument("--lovasz_w", type=float, default=0.0,
               help="weight for present-class, per-image Lovasz-Softmax loss")
p.add_argument("--lovasz_max_pixels", type=int, default=131072,
               help="uniform pixels per image used to approximate Lovasz on large crops")
p.add_argument("--init_checkpoint", default="",
               help="optional segmentation state_dict used to initialize fine-tuning")
p.add_argument("--deterministic_workers", action="store_true",
               help="seed each data-loader worker deterministically from --seed")
p.add_argument("--cldice_w", type=float, default=0.0, help="weight of clDice topology loss (0 = off)")
p.add_argument("--cldice_classes", default="1,3", help="elongated phases: 1 Al3Ni rods, 3 eutectic Si needles")
p.add_argument("--cldice_iters", type=int, default=10, help="soft-skeleton iterations (~max half-thickness in px)")
p.add_argument("--cldice_pool", type=int, default=1, help="avg-pool factor before skeletonizing (memory)")
p.add_argument("--grad_ckpt", action="store_true", help="encoder gradient checkpointing (~40% slower, much less memory)")
p.add_argument("--ema", type=float, default=0.0, help="EMA decay for a shadow copy of the weights (0 = off)")
p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--no_valid_eval", action="store_true",
               help="do not inspect official validation masks during this fixed-epoch CV run")
p.add_argument("--amp_init_scale", type=float, default=1024,
               help="initial FP16 GradScaler scale (P100-safe default)")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--input", default="gray3", choices=["gray3", "feat3", "zcdf3"],
               help="gray3, feat3: [gray, Hessian ridge, high-pass], or zcdf3: [raw, full-image z-score, full-image CDF]")
p.add_argument("--ema_start",type=int,default=12000)
p.add_argument("--bn_batches",type=int,default=300)
p.add_argument("--bn_seed",type=int,default=819326)
p.add_argument("--encoder_lr_mult",type=float,default=.2)
p.add_argument("--freeze_bn_stats",action="store_true")
p.add_argument("--swa_start",type=int,default=840)
p.add_argument("--swa_every",type=int,default=120)
args = p.parse_args()
args.amp_dtype = "float16"

os.makedirs(args.out, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_pairs = [pm for s in args.train_splits.split(",") for pm in list_split(args.data, s)]
valid_pairs = (list_split(args.data, "valid")
               if "valid" not in args.train_splits and not args.no_valid_eval else [])
print(f"train {len(train_pairs)} valid {len(valid_pairs)}", flush=True)

args.feat_stats = None
if args.input == "feat3":  # standardization constants from training images only
    rs, hs = [], []
    for ip, _ in train_pairs:
        g = load_gray(ip)
        ridge, hp = morph_features(resize_gray(g, args.target_width / g.shape[1] if args.target_width else args.scale))
        rs.append(ridge[::4, ::4].ravel()); hs.append(hp[::4, ::4].ravel())
    rs, hs = np.concatenate(rs), np.concatenate(hs)
    args.feat_stats = [float(rs.mean()), float(rs.std()), float(hs.mean()), float(hs.std())]
    print("feat_stats", args.feat_stats, flush=True)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
CLDICE_CLASSES = [int(c) for c in args.cldice_classes.split(",")]

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


ds = CropDS(train_pairs, args.iters * args.bs)
dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True,
                                 worker_init_fn=worker_init, persistent_workers=True, drop_last=True)
valid_data = [(os.path.basename(i), load_gray(i), load_mask(m)) for i, m in valid_pairs]

model = build_model(args.arch, args.encoder, pretrained=False).cuda()
if args.init_checkpoint:
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    print(f"initialized from segmentation checkpoint {args.init_checkpoint}", flush=True)
if args.grad_ckpt:  # trade compute for memory so larger crops fit (timm encoders only)
    for obj in (model.encoder, getattr(model.encoder, "model", None)):
        if hasattr(obj, "set_grad_checkpointing"):
            obj.set_grad_checkpointing(True)
            break
    else:
        raise SystemExit(f"{type(model.encoder).__name__} has no gradient checkpointing; use a tu-* encoder")
ema_model = None
import copy
# EMA is initialized only at the fixed late-training boundary; it never affects the optimizer trajectory.
encoder_params=list(model.encoder.parameters());encoder_ids={id(p) for p in encoder_params};head_params=[p for p in model.parameters() if id(p) not in encoder_ids]
opt = torch.optim.AdamW([{ "params": encoder_params, "lr":args.lr*args.encoder_lr_mult }, { "params":head_params, "lr":args.lr }], weight_decay=args.wd)
swa_model=None;swa_count=0
warm = 168
steps = args.iters // args.accum  # optimizer steps (sched.step() runs once per optimizer step)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(it, steps) / steps)))


def dice_loss(logits, y):
    prob = logits.float().softmax(1)
    oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (prob * oh).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


# ---- clDice (Shit et al., CVPR 2021): overlap of soft skeletons, rewards preserving the
# elongated/connected shape of a phase rather than just its area.
def _soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_open(x):
    return F.max_pool2d(_soft_erode(x), 3, 1, 1)


def soft_skeleton(x, iters):
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, y, classes, iters, pool=1):
    prob = logits.float().softmax(1)[:, classes]
    oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()[:, classes]
    if pool > 1:  # compute at lower resolution to bound memory (skeletons need `iters` pooling passes)
        prob, oh = F.avg_pool2d(prob, pool), F.avg_pool2d(oh, pool)
    sp = soft_skeleton(prob, iters)
    with torch.no_grad():
        sg = soft_skeleton(oh, iters)
    tprec = ((sp * oh).sum((0, 2, 3)) + 1) / (sp.sum((0, 2, 3)) + 1)
    tsens = ((sg * prob).sum((0, 2, 3)) + 1) / (sg.sum((0, 2, 3)) + 1)
    return 1 - (2 * tprec * tsens / (tprec + tsens)).mean()


def evaluate(tta=False, net=None):
    net = net or model
    net.eval()
    rows = []
    for name, g, gt in valid_data:
        member = make_member(net, args.norm, args.scale, args.input, args.feat_stats, args.target_width)
        pred = predict_probs([member], g, tta=tta).argmax(0).cpu().numpy()
        a, b, iou = image_miou(pred, gt)
        rows.append((name, a, b, iou))
    model.train()
    return rows


@torch.no_grad()
def ema_update(decay):
    for e,m in zip(ema_model.parameters(),model.parameters()):e.mul_(decay).add_(m,alpha=1-decay)
    for e,m in zip(ema_model.buffers(),model.buffers()):e.copy_(m)


log = open(os.path.join(args.out, "log.txt"), "a")
model.train()
if args.freeze_bn_stats:
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
scaler = torch.amp.GradScaler("cuda", init_scale=args.amp_init_scale)
t0 = time.time()
optimizer_updates=0;skipped_updates=0;ema_updates=0
run = 0.0
for it, (x, y) in enumerate(dl):
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x)
    loss = F.cross_entropy(logits.float(), y) + args.dice_w * dice_loss(logits.float(), y)
    if args.lovasz_w:
        loss = loss + args.lovasz_w * imagewise_lovasz_loss(
            logits, y, args.lovasz_max_pixels)
    if args.cldice_w:
        loss = loss + args.cldice_w * cldice_loss(logits, y, CLDICE_CLASSES, args.cldice_iters, args.cldice_pool)
    scaler.scale(loss / args.accum).backward()
    if (it + 1) % args.accum == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale=scaler.get_scale()
        scaler.step(opt); scaler.update(); sched.step()
        stepped=scaler.get_scale()>=old_scale
        optimizer_updates+=int(stepped);skipped_updates+=int(not stepped)
        opt.zero_grad(set_to_none=True)
        if args.ema and (it+1)>=args.ema_start and stepped:
            if ema_model is None:
                ema_model=copy.deepcopy(model).eval()
                for param in ema_model.parameters():param.requires_grad_(False)
            else:
                ema_update(args.ema);ema_updates+=1
    if stepped and (it+1)>=args.swa_start and (it+1)%args.swa_every==0:
        if swa_model is None:swa_model=copy.deepcopy(model).eval();swa_count=1
        else:
            with torch.no_grad():
                for avg,current in zip(swa_model.parameters(),model.parameters()):avg.add_((current-avg)/(swa_count+1))
                for avg,current in zip(swa_model.buffers(),model.buffers()):avg.copy_(current)
            swa_count+=1
    run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
    if (it + 1) % 100 == 0:
        msg = (f"it {it+1} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
               f"scale {scaler.get_scale():.0f} {time.time()-t0:.0f}s "
               f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if valid_data and ((it + 1) % args.eval_every == 0 or it + 1 == args.iters):
        rows = evaluate()
        msg = f"[valid noTTA] it {it+1} mIoU {np.mean([r[1] for r in rows])*100:.2f} (absent=1: {np.mean([r[2] for r in rows])*100:.2f})"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
if swa_model is not None:
    torch.save(swa_model.state_dict(),os.path.join(args.out,"swa.pt"))
json.dump({"optimizer_updates":optimizer_updates,"skipped_updates":skipped_updates,"ema_updates":ema_updates,"swa_snapshots":swa_count,"freeze_bn_stats":args.freeze_bn_stats,"valid_used_for_training":False,"elapsed_seconds":time.time()-t0},open(os.path.join(args.out,"training_stats.json"),"w"),indent=2)
if ema_model is not None:
    assert ema_updates>0
    torch.save(ema_model.state_dict(), os.path.join(args.out, "ema.pt"))
    # Calibrate both models on identical fold-specific, seed-independent training batches.
    def calibrate_bn(net):
        random.seed(args.bn_seed);np.random.seed(args.bn_seed);torch.manual_seed(args.bn_seed)
        net.eval()
        for module in net.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats();module.momentum=None;module.train()
        calibration=torch.utils.data.DataLoader(CropDS(train_pairs,args.bn_batches*args.bs),batch_size=args.bs,num_workers=0,drop_last=True)
        with torch.no_grad():
            for xx,_ in calibration:
                with torch.autocast("cuda",dtype=torch.float16):net(xx.cuda(non_blocking=True))
        net.eval()
    if args.freeze_bn_stats:
        log.close();raise SystemExit(0)
    calibrate_bn(model);torch.save(model.state_dict(),os.path.join(args.out,"plain_bn.pt"))
    calibrate_bn(ema_model);torch.save(ema_model.state_dict(),os.path.join(args.out,"ema_bn.pt"))
    json.dump({"optimizer_updates":optimizer_updates,"skipped_updates":skipped_updates,"ema_updates":ema_updates,"ema_start":args.ema_start,"ema_decay":args.ema,"bn_batches":args.bn_batches,"bn_seed":args.bn_seed,"valid_used_for_training":False,"elapsed_seconds":time.time()-t0},open(os.path.join(args.out,"training_stats.json"),"w"),indent=2)


def final_eval(net, tag, fname):
    rows = evaluate(tta=True, net=net)
    per_class = np.nanmean(np.stack([r[3] for r in rows]), 0) * 100
    res = {"miou_tta": float(np.mean([r[1] for r in rows]) * 100),
           "miou_tta_absent1": float(np.mean([r[2] for r in rows]) * 100),
           "per_class_iou": per_class.round(2).tolist(),
           "per_image": {r[0]: round(float(r[1]) * 100, 2) for r in rows}}
    json.dump(res, open(os.path.join(args.out, fname), "w"), indent=2)
    msg = (f"[valid TTA final{tag}] mIoU {res['miou_tta']:.2f} "
           f"(absent=1: {res['miou_tta_absent1']:.2f}) per-class {res['per_class_iou']}")
    print(msg, flush=True); log.write(msg + "\n"); log.flush()


if valid_data:
    final_eval(model, "", "valid.json")
    if ema_model is not None:
        final_eval(ema_model, " EMA", "valid_ema.json")
