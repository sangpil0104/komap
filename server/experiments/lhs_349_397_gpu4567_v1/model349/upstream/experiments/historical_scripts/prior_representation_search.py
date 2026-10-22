#!/usr/bin/env python3
"""Compare z-score and percentile-domain adaptive intensity priors."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import gaussian_prior
from tent_style_search import adapt_model

COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def percentile_image(gray):
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind="stable")
    result = np.empty(flat.shape, dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return result.reshape(gray.shape)


def fit_percentile_stats(data_dir):
    values = [[] for _ in range(4)]
    for image_path, mask_path in collect_split(Path(data_dir), "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        q = percentile_image(gray)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for cid, color in enumerate(COLORS): ids[np.all(mask == color, axis=2)] = cid
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for cid in range(4):
            current = q[ids == cid]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[cid].append(current[take])
    return [(float(np.concatenate(v).mean()), max(float(np.concatenate(v).std()), .02)) for v in values]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    data_dir = Path(args.data_dir); train_stats = []
    for p, _ in collect_split(data_dir, "train"):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32); train_stats.append((float(g.mean()), float(g.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats = []
    for p, mask_path in collect_split(data_dir, "train"):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32); z = (g - g.mean()) / max(float(g.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB")); ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for cid, color in enumerate(COLORS): ids[np.all(mask == color, axis=2)] = cid
        rng = np.random.default_rng(sum(p.name.encode("utf-8")))
        for cid in range(4):
            current = z[ids == cid]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                if len(z_stats) <= cid: z_stats.append([])
                z_stats[cid].append(current[take])
    z_stats = [(float(np.concatenate(v).mean()), max(float(np.concatenate(v).std()), .15)) for v in z_stats]
    q_stats = fit_percentile_stats(data_dir)
    pair = load_aspp(args.pair_checkpoint, device); items = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8); gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std), 768, 576, device, amp, 1, "d4", "gaussian")
        high_weight = .75 if float(gray.mean()) < 165.0 else .55
        high = .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high
        base = MATRIX * (1.0 - high_weight - .15)
        mixed = np.asarray([sum(base[c, m] * loaded[f"p{m}"].astype(np.float32)[c] for m in range(3)) + high_weight * high[c] + .15 * pair_value[c] for c in range(4)], np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        mean = float(gray.mean()); z = (gray - mean) / max(float(gray.std()), 1e-5); q = percentile_image(gray)
        adaptive_z=[]; adaptive_q=[]
        for cid in range(4):
            w=np.maximum(mixed[cid],1e-7)**2.0
            mz=float((w*z).sum()/w.sum()); sz=float(np.sqrt((w*(z-mz)**2).sum()/w.sum())); adaptive_z.append((mz,max(sz,.15)))
            mq=float((w*q).sum()/w.sum()); sq=float(np.sqrt((w*(q-mq)**2).sum()/w.sum())); adaptive_q.append((mq,max(sq,.02)))
        items.append({"mixed":mixed,"z_prior":gaussian_prior(z,z_stats),"q_prior":gaussian_prior(q,q_stats),
                      "z_adapt":gaussian_prior(z,adaptive_z),"q_adapt":gaussian_prior(q,adaptive_q),"mean":mean,
                      "target":decode_mask(Image.open(mask_path).convert("RGB")),"name":image_path.name})
        del adapted,raw_high,style_high; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
        print(f"cached {image_path.name}",flush=True)

    records=[]
    for q_weight in (0.0,.15,.30,.50,.70,.85,1.0):
        predictions=[];targets=[]
        for item in items:
            prior=(1-q_weight)*item["z_prior"]+q_weight*item["q_prior"]
            delta=(1-q_weight)*(item["z_adapt"][0]-item["z_prior"][0])+q_weight*(item["q_adapt"][0]-item["q_prior"][0])
            prior[0]+= (3.0 if item["mean"]<160 else 7.0)*delta
            logits=np.log(np.maximum(item["mixed"],1e-7))+.09*prior; logits[2]+=.40; logits[3]+=-1.0 if item["mean"]<165 else -.30
            prediction=cleanup_primary(logits.argmax(0).astype(np.uint8),logits,128)
            if float((prediction==1).mean())<=.01:
                mask=prediction==1; alt=logits.copy(); alt[1]=-np.inf; prediction[mask]=alt[:,mask].argmax(0).astype(np.uint8)
            predictions.append(prediction);targets.append(item["target"])
        records.append({"q_weight":q_weight,**metrics(predictions,targets)})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    best=max(records,key=lambda x:x["image_mean_miou"]);out=Path(args.output_json);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"best":best,"records":records},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"best":best},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",required=True);p.add_argument("--cache-dir",required=True);p.add_argument("--pair-checkpoint",required=True);p.add_argument("--high-checkpoint",required=True);p.add_argument("--output-json",required=True);main(p.parse_args())
