#!/usr/bin/env python3
"""Calibration search for the current 768-context ensemble cache."""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
from aspp768_addition import BIASES, MATRIX, cleanup_primary
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


def load_cached(data_dir, cache_dir, high_weight):
    result=[]
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded=np.load(Path(cache_dir)/(image_path.stem+".npz"))
        values=[loaded[f"p{i}"].astype(np.float32) for i in range(4)]
        prior=loaded["prior"].astype(np.float32)
        base=MATRIX*(1.0-high_weight)
        mixed=np.asarray([sum(base[c,m]*values[m][c] for m in range(3))+high_weight*values[3][c]
                          for c in range(4)],np.float32)
        mixed/=np.maximum(mixed.sum(0,keepdims=True),1e-7)
        result.append((np.log(np.maximum(mixed,1e-7)),prior,decode_mask(Image.open(mask_path).convert("RGB")),image_path.name))
    return result


def evaluate(cached, alpha, biases, threshold):
    predictions=[]; targets=[]
    bias=np.asarray(biases,dtype=np.float32)[:,None,None]
    for base,prior,target,_ in cached:
        logits=base+alpha*prior+bias
        predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8),logits,threshold))
        targets.append(target)
    return metrics(predictions,targets)


def main(args):
    cached=load_cached(args.data_dir,args.cache_dir,args.high_weight)
    candidates=[]
    for alpha in (0.0,0.03,0.06,0.09,0.12,0.15,0.20):
        for al_bias in (0.0,0.2,0.4,0.6,0.8):
            for eut_bias in (-1.0,-0.8,-0.65,-0.55,-0.45,-0.3,-0.1):
                biases=[0.0,0.0,al_bias,eut_bias]
                value=evaluate(cached,alpha,biases,args.threshold)
                candidates.append({"alpha":alpha,"biases":biases,"threshold":args.threshold,**value})
    best=max(candidates,key=lambda x:x["image_mean_miou"])
    # Coordinate refinement around the best point with a wider local grid.
    current=(float(best["alpha"]),list(best["biases"]))
    history=[best]
    for _ in range(3):
        improved=False
        for kind,values in (
            ("alpha",(0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.18,0.22)),
            ("al3ni",(-0.2,-0.1,0.0,0.1,0.2,0.3)),
            ("primary",(-0.2,-0.1,0.0,0.1,0.2)),
            ("al",(-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),
            ("eut",(-1.2,-1.0,-0.9,-0.8,-0.7,-0.6,-0.55,-0.5,-0.4,-0.3,-0.2,-0.1,0.0)),
        ):
            local=[]
            for value in values:
                alpha=current[0]; biases=current[1].copy()
                if kind=="alpha": alpha=value
                elif kind=="al3ni": biases[0]=value
                elif kind=="primary": biases[1]=value
                elif kind=="al": biases[2]=value
                else: biases[3]=value
                result=evaluate(cached,alpha,biases,args.threshold)
                local.append({"alpha":alpha,"biases":biases,"threshold":args.threshold,**result})
            selected=max(local,key=lambda x:x["image_mean_miou"])
            if selected["image_mean_miou"]>history[-1]["image_mean_miou"]+1e-10: improved=True
            current=(selected["alpha"],selected["biases"]); history.append(selected)
            print(json.dumps({"kind":kind,"alpha":current[0],"biases":current[1],
                              "image_mean_miou":selected["image_mean_miou"],"image_miou_compat":selected["image_miou_compat"]},ensure_ascii=False),flush=True)
        if not improved: break
    output=Path(args.output_json); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"grid_best":best,"coordinate_best":history[-1],"history":history},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"grid_best":{k:best[k] for k in ("alpha","biases","threshold","image_miou_compat","image_mean_miou")},
                      "coordinate_best":{k:history[-1][k] for k in ("alpha","biases","threshold","image_miou_compat","image_mean_miou")}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True); p.add_argument("--cache-dir",required=True)
    p.add_argument("--output-json",required=True); p.add_argument("--high-weight",type=float,default=.6)
    p.add_argument("--threshold",type=int,default=128); main(p.parse_args())
