#!/usr/bin/env python3
"""Fine search of Al/Eutectic biases on the current percentile-prior route."""
from __future__ import annotations
import argparse, gc, itertools, json
from pathlib import Path
import numpy as np, torch
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
    m=build_model("aspp",pretrained=False,checkpoint=None).to(device);m.load_state_dict(torch.load(path,map_location="cpu")["model"]);m.eval();return m

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");amp=device.type=="cuda";data=Path(args.data_dir)
    ts=[]
    for p,_ in collect_split(data,"train"):
        g=np.asarray(Image.open(p).convert("L"),np.float32);ts.append((float(g.mean()),float(g.std())))
    tmean=float(np.mean([x[0] for x in ts]));tstd=float(np.mean([x[1] for x in ts]));zs=fit_global_stats(data);qs=fit_percentile_stats(data);pair=load_aspp(args.pair_checkpoint,device);items=[]
    for ip,mp in collect_split(data,"valid"):
        l=np.load(Path(args.cache_dir)/f"{ip.stem}.npz");raw=np.asarray(Image.open(ip).convert("RGB"),np.uint8);gray=raw[...,0].astype(np.float32);pp=predict_probabilities(pair,ip,768,576,device,amp,1,"d4","gaussian");model=adapt_model(args.high_checkpoint,raw,device,amp,5,1e-5);rh=predict_array(model,raw,768,576,device,amp,1,"d4","gaussian");sh=predict_array(model,transform_image(raw,"standardize",tmean,tstd),768,576,device,amp,1,"d4","gaussian");mean=float(gray.mean());hw=.75 if mean<165 else .55;high=.25*rh+.75*sh if float(gray.std())>=50 else rh;base=MATRIX*(1-hw-.15);mixed=np.asarray([sum(base[c,m]*l[f"p{m}"].astype(np.float32)[c] for m in range(3))+hw*high[c]+.15*pp[c] for c in range(4)],np.float32);mixed/=np.maximum(mixed.sum(0,keepdims=True),1e-7);z=(gray-mean)/max(float(gray.std()),1e-5);q=percentile_image(gray);az=[];aq=[]
        for cid in range(4):
            w=np.maximum(mixed[cid],1e-7)**2;mz=float((w*z).sum()/w.sum());sz=float(np.sqrt((w*(z-mz)**2).sum()/w.sum()));mq=float((w*q).sum()/w.sum());sq=float(np.sqrt((w*(q-mq)**2).sum()/w.sum()));az.append((mz,max(sz,.15)));aq.append((mq,max(sq,.02)))
        prior=.85*gaussian_prior(z,zs)+.15*gaussian_prior(q,qs);delta=.85*(gaussian_prior(z,az)[0]-gaussian_prior(z,zs)[0])+.15*(gaussian_prior(q,aq)[0]-gaussian_prior(q,qs)[0]);prior[0]+=(3 if mean<160 else 7)*delta;items.append({"mixed":mixed,"prior":prior,"mean":mean,"target":decode_mask(Image.open(mp).convert("RGB"))});del model,rh,sh;gc.collect();torch.cuda.empty_cache();print(f"cached {ip.name}",flush=True)
    records=[]
    for al_bias,low_eut,high_eut,bth in itertools.product((.25,.35,.45,.55,.65),(-1.3,-1.0,-.7),(-.5,-.3,-.1),(160.,165.,170.)):
        ps=[];ys=[]
        for x in items:
            logits=np.log(np.maximum(x["mixed"],1e-7))+.09*x["prior"];logits[2]+=al_bias;logits[3]+=high_eut if x["mean"]>=bth else low_eut;pred=cleanup_primary(logits.argmax(0).astype(np.uint8),logits,128)
            if float((pred==1).mean())<=.01:
                mask=pred==1;alt=logits.copy();alt[1]=-np.inf;pred[mask]=alt[:,mask].argmax(0).astype(np.uint8)
            ps.append(pred);ys.append(x["target"])
        records.append({"al_bias":al_bias,"low_eut_bias":low_eut,"high_eut_bias":high_eut,"bias_threshold":bth,**metrics(ps,ys)})
    best=max(records,key=lambda x:x["image_mean_miou"]);out=Path(args.output_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({"best":best,"records":records},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"best":best},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",required=True);p.add_argument("--cache-dir",required=True);p.add_argument("--pair-checkpoint",required=True);p.add_argument("--high-checkpoint",required=True);p.add_argument("--output-json",required=True);main(p.parse_args())
