#!/usr/bin/env python3
"""Train a train-only pixel residual refiner for the 768 ASPP model."""
from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

from aspp768_addition import BIASES, MATRIX, cleanup_primary
from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask


def pixel_features(image_path, probabilities):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = np.dot(image, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    local_mean = uniform_filter(gray, size=15, mode="reflect")
    local_sq = uniform_filter(gray * gray, size=15, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 1e-6))
    gradient = np.sqrt(sobel(gray, axis=0, mode="reflect") ** 2 + sobel(gray, axis=1, mode="reflect") ** 2)
    gradient /= max(float(np.percentile(gradient, 99.0)), 1e-6)
    logp = np.log(np.maximum(probabilities, 1e-7))
    entropy = -(probabilities * logp).sum(axis=0)
    values = np.concatenate([
        logp,
        image.transpose(2, 0, 1),
        gray[None], local_mean[None], local_std[None], gradient[None], entropy[None],
    ], axis=0)
    return values.transpose(1, 2, 0).reshape(-1, values.shape[0]).astype(np.float32)


class PixelRefiner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


def infer_train_features(args, device, amp):
    model = build_model("aspp", pretrained=False, checkpoint=args.high_base_checkpoint).to(device)
    model.load_state_dict(torch.load(args.high_checkpoint, map_location="cpu")["model"])
    model.eval()
    all_x, all_base, all_y = [], [], []
    rng = np.random.default_rng(args.seed)
    for image_path, mask_path in collect_split(args.data_dir, "train"):
        probabilities = predict_probabilities(model, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        features = pixel_features(image_path, probabilities)
        base = np.log(np.maximum(probabilities, 1e-7)).transpose(1, 2, 0).reshape(-1, 4).astype(np.float32)
        labels = target.reshape(-1)
        selected = []
        for class_id in range(4):
            indices = np.flatnonzero(labels == class_id)
            if len(indices):
                take = min(args.samples_per_class, len(indices))
                selected.append(rng.choice(indices, take, replace=False))
        indices = np.concatenate(selected); rng.shuffle(indices)
        all_x.append(features[indices]); all_base.append(base[indices]); all_y.append(labels[indices])
        print(f"cached train {image_path.name} samples={len(indices)}", flush=True)
    return (np.concatenate(all_x), np.concatenate(all_base), np.concatenate(all_y))


def train_refiner(x, base, y, args, device):
    model = PixelRefiner(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.refiner_lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    x_tensor = torch.from_numpy(x); base_tensor = torch.from_numpy(base); y_tensor = torch.from_numpy(y).long()
    for epoch in _local_progress(range(args.refiner_epochs), 'pixel_refiner_768'):
        order = rng.permutation(len(y)); total = 0.0; count = 0
        model.train()
        for start in range(0, len(order), args.batch_size):
            index = torch.from_numpy(order[start:start + args.batch_size]).long()
            xb = x_tensor[index].to(device); bb = base_tensor[index].to(device); yb = y_tensor[index].to(device)
            delta = model(xb)
            loss = F.cross_entropy(bb + delta, yb)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.item()) * len(index); count += len(index)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"refiner epoch={epoch + 1} loss={total / max(count, 1):.5f}", flush=True)
    return model


@torch.no_grad()
def delta_for(model, image_path, probabilities, device, batch_size):
    features = torch.from_numpy(pixel_features(image_path, probabilities)).to(device)
    model.eval(); values=[]
    for start in range(0, len(features), batch_size):
        values.append(model(features[start:start + batch_size]).cpu().numpy())
    return np.concatenate(values).reshape(probabilities.shape[1], probabilities.shape[2], 4).transpose(2, 0, 1)


def evaluate_refiner(refiner, valid_cache, args, device):
    records=[]
    for refiner_alpha in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0):
        for high_weight in (0.4, 0.5, 0.6, 0.7):
            predictions=[]; targets=[]
            for image_path, values, prior, target in valid_cache:
                refined_logits = np.log(np.maximum(values[3], 1e-7)) + refiner_alpha * delta_for(
                    refiner, image_path, values[3], device, args.infer_batch_size)
                refined = np.exp(refined_logits - refined_logits.max(0, keepdims=True))
                refined /= np.maximum(refined.sum(0, keepdims=True), 1e-7)
                sources = [values[0], values[1], values[2], refined]
                base = MATRIX * (1.0 - high_weight)
                mixed = np.asarray([sum(base[c, m] * sources[m][c] for m in range(3))
                                    + high_weight * sources[3][c] for c in range(4)], np.float32)
                mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
                logits = np.log(np.maximum(mixed, 1e-7)) + args.prior_alpha * prior
                logits += np.asarray(args.biases, np.float32)[:, None, None]
                prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, args.cleanup_threshold)
                predictions.append(prediction); targets.append(target)
            cm_total=np.zeros((4,4),np.int64); image_scores=[]
            for prediction,target in zip(predictions,targets):
                cm=np.bincount(target.reshape(-1)*4+prediction.reshape(-1),minlength=16).reshape(4,4)
                cm_total+=cm; inter=np.diag(cm).astype(float); union=cm.sum(0)+cm.sum(1)-inter
                image_scores.append(float(np.mean(inter/np.maximum(union,1.0))))
            inter=np.diag(cm_total).astype(float); union=cm_total.sum(0)+cm_total.sum(1)-inter
            iou=inter/np.maximum(union,1.0)
            records.append({"refiner_alpha":refiner_alpha,"high_weight":high_weight,
                            "image_miou_compat":float(iou.mean()),"image_mean_miou":float(np.mean(image_scores)),
                            "image_class_iou":iou.tolist()})
            print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    return records


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp=device.type=="cuda"
    x,base,y=infer_train_features(args,device,amp)
    print(f"train features={x.shape} labels={y.shape}",flush=True)
    refiner=train_refiner(x,base,y,args,device)
    valid_cache=[]
    cache_dir=Path(args.valid_cache_dir)
    for image_path,mask_path in collect_split(args.data_dir,"valid"):
        loaded=np.load(cache_dir/(image_path.stem+".npz"))
        values=[loaded[f"p{i}"].astype(np.float32) for i in range(4)]
        valid_cache.append((image_path,values,loaded["prior"].astype(np.float32),decode_mask(Image.open(mask_path).convert("RGB"))))
    records=evaluate_refiner(refiner,valid_cache,args,device)
    best=max(records,key=lambda x:x["image_mean_miou"])
    out=Path(args.output_json); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":refiner.state_dict(),"input_dim":x.shape[1]},out.with_suffix(".pt"))
    out.write_text(json.dumps({"best":best,"records":records},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"best":best},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True); p.add_argument("--high-checkpoint",required=True)
    p.add_argument("--high-base-checkpoint",required=True); p.add_argument("--valid-cache-dir",required=True)
    p.add_argument("--output-json",required=True); p.add_argument("--samples-per-class",type=int,default=2500)
    p.add_argument("--refiner-epochs",type=int,default=20); p.add_argument("--refiner-lr",type=float,default=1e-3)
    p.add_argument("--batch-size",type=int,default=32768); p.add_argument("--infer-batch-size",type=int,default=65536)
    p.add_argument("--prior-alpha",type=float,default=.12); p.add_argument("--biases",type=float,nargs=4,default=[0.,0.,.4,-.55])
    p.add_argument("--cleanup-threshold",type=int,default=128); p.add_argument("--seed",type=int,default=123)
    main(p.parse_args())
