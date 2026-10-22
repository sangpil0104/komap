#!/usr/bin/env python3
"""Evaluate denser high-resolution sliding-window overlap in the best route."""

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
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval(); return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    data_dir = Path(args.data_dir); train=[]
    for p,_ in collect_split(data_dir,"train"):
        g=np.asarray(Image.open(p).convert("L"),np.float32); train.append((float(g.mean()),float(g.std())))
    train_mean=float(np.mean([x[0] for x in train])); train_std=float(np.mean([x[1] for x in train]))
    z_stats=fit_global_stats(data_dir); q_stats=fit_percentile_stats(data_dir); pair=load_aspp(args.pair_checkpoint,device)
    entries=[]
    for ip,mp in collect_split(data_dir,"valid"):
        loaded=np.load(Path(args.cache_dir)/f"{ip.stem}.npz"); raw=np.asarray(Image.open(ip).convert("RGB"),np.uint8); gray=raw[...,0].astype(np.float32)
        pp=predict_probabilities(pair,ip,768,576,device,amp,1,"d4","gaussian")
        entries.append({"ip":ip,"raw":raw,"gray":gray,"src":[loaded[f"p{i}"].astype(np.float32) for i in range(3)],"prior":loaded["prior"].astype(np.float32),"pair":pp,"target":decode_mask(Image.open(mp).convert("RGB"))})
        print(f"cached {ip.name}",flush=True)
    records=[]
    for stride in args.strides:
        preds=[];targets=[]
        for e in entries:
            model=adapt_model(args.high_checkpoint,e["raw"],device,amp,5,1e-5)
            raw_high=predict_array(model,e["raw"],768,stride,device,amp,1,"d4","gaussian")
            style_high=predict_array(model,transform_image(e["raw"],"standardize",train_mean,train_std),768,stride,device,amp,1,"d4","gaussian")
            gray=e["gray"]; mean=float(gray.mean()); high=.25*raw_high+.75*style_high if float(gray.std())>=50 else raw_high; hw=.75 if mean<165 else .55; base=MATRIX*(1-hw-.15)
            mixed=np.asarray([sum(base[c,m]*e["src"][m][c] for m in range(3))+hw*high[c]+.15*e["pair"][c] for c in range(4)],np.float32);mixed/=np.maximum(mixed.sum(0,keepdims=True),1e-7)
            z=(gray-mean)/max(float(gray.std()),1e-5);q=percentile_image(gray); az=[];aq=[]
            for cid in range(4):
                w=np.maximum(mixed[cid],1e-7)**2; mz=float((w*z).sum()/w.sum()); sz=float(np.sqrt((w*(z-mz)**2).sum()/w.sum())); mq=float((w*q).sum()/w.sum()); sq=float(np.sqrt((w*(q-mq)**2).sum()/w.sum())); az.append((mz,max(sz,.15)));aq.append((mq,max(sq,.02)))
            prior=.85*gaussian_prior(z,z_stats)+.15*gaussian_prior(q,q_stats); dz=.85*(gaussian_prior(z,az)[0]-gaussian_prior(z,z_stats)[0])+.15*(gaussian_prior(q,aq)[0]-gaussian_prior(q,q_stats)[0]); prior[0]+=(3 if mean<160 else 7)*dz
            logits=np.log(np.maximum(mixed,1e-7))+.09*prior;logits[2]+=.40;logits[3]+=-1 if mean<165 else -.30;pred=cleanup_primary(logits.argmax(0).astype(np.uint8),logits,128)
            if float((pred==1).mean())<=.01:
                mask=pred==1;alt=logits.copy();alt[1]=-np.inf;pred[mask]=alt[:,mask].argmax(0).astype(np.uint8)
            preds.append(pred);targets.append(e["target"]);del model,raw_high,style_high;gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
        row={"stride":stride,**metrics(preds,targets)};records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    best=max(records,key=lambda x:x["image_mean_miou"]);out=Path(args.output_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({"best":best,"records":records},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"best":best},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",required=True);p.add_argument("--cache-dir",required=True);p.add_argument("--pair-checkpoint",required=True);p.add_argument("--high-checkpoint",required=True);p.add_argument("--output-json",required=True);p.add_argument("--strides",type=int,nargs="+",default=[384]);main(p.parse_args())
