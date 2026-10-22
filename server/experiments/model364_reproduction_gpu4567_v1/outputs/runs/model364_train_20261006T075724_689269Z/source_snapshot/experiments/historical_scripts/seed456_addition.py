#!/usr/bin/env python3
"""Evaluate the independent 768 seed as a fifth source."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from PIL import Image
from advanced_inference import predict_probabilities
from aspp768_addition import BIASES, MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp=device.type=="cuda"
    model=build_model("aspp",pretrained=False,checkpoint=args.base_checkpoint).to(device)
    model.load_state_dict(torch.load(args.seed_checkpoint,map_location="cpu")["model"]); model.eval()
    cache=[]
    for image_path,mask_path in collect_split(args.data_dir,"valid"):
        loaded=np.load(Path(args.base_cache)/(image_path.stem+".npz"))
        values=[loaded[f"p{i}"].astype(np.float32) for i in range(4)]
        seed=predict_probabilities(model,image_path,768,576,device,amp,1,"d4","gaussian")
        cache.append((values,seed,loaded["prior"].astype(np.float32),decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}",flush=True)
    records=[]
    for seed_weight in (0.0,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.40):
        predictions=[];targets=[]
        for values,seed,prior,target in cache:
            # Start with the model112 weights: old/new/aspp total .4 and
            # seed123 highres .6. Scale that mixture before adding seed456.
            base=MATRIX*.4*(1.0-seed_weight)
            mixed=np.asarray([sum(base[c,m]*values[m][c] for m in range(3))
                              + .6*(1.0-seed_weight)*values[3][c]
                              + seed_weight*seed[c] for c in range(4)],np.float32)
            mixed/=np.maximum(mixed.sum(0,keepdims=True),1e-7)
            logits=np.log(np.maximum(mixed,1e-7))+.12*prior+BIASES[:,None,None]
            # The best calibration search found Al bias +0.40, replacing
            # model112's +0.20 while keeping the other biases unchanged.
            logits[2]+=.20
            predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8),logits,128));targets.append(target)
        records.append({"seed456_weight":seed_weight,**metrics(predictions,targets)})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    best=max(records,key=lambda x:x["image_mean_miou"]); output=Path(args.output_json);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"best":best,"records":records},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"best":best},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",required=True);p.add_argument("--base-cache",required=True)
    p.add_argument("--base-checkpoint",required=True);p.add_argument("--seed-checkpoint",required=True);p.add_argument("--output-json",required=True);main(p.parse_args())
