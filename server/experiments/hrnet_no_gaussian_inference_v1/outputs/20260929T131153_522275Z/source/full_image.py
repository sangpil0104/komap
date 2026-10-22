"""Test full-image context with the same frozen HRNet checkpoint and normalization."""
import argparse
from contextlib import nullcontext
import gc
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import ROOT,BASELINE,DEADLINE,require_environment,data_rows,save_json,reserve_result,score_image,finalize,update_comparison
from inference import VIEWS,transform
sys.path.insert(0,str(ROOT/'vendor'))
from komap_seg.checkpoint import load_model
from komap_seg.dataset import build_eval_transform


@torch.inference_mode()
def predict(model,image,device,scale=2.,tta='flip4',precision='fp32'):
    total=None
    for view in VIEWS[tta]:
        if time.time()>=DEADLINE:raise TimeoutError('Requested deadline reached')
        x=transform(image,view).unsqueeze(0).to(device);h,w=x.shape[-2:]
        size=(round(h*scale),round(w*scale))
        if scale!=1:x=F.interpolate(x,size=size,mode='bicubic',align_corners=False)
        H,W=x.shape[-2:];x=F.pad(x,(0,(-W)%32,0,(-H)%32),mode='reflect')
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
        with ctx:logits=model(x)
        p=logits.float()[...,:H,:W].softmax(1)
        if (H,W)!=(h,w):p=F.interpolate(p,size=(h,w),mode='bilinear',align_corners=False)
        p=transform(p[0],view,inverse=True).cpu()
        total=p if total is None else total+p
    return (total/len(VIEWS[tta])).numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7);p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    require_environment();torch.set_num_threads(1);torch.set_grad_enabled(False)
    device=torch.device(f'cuda:{args.gpu}');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.36,device)
    arms=json.loads(args.plan.read_text());run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    save_json(run/'plan.json',arms);print('RUN:',run,flush=True)
    model=load_model(Path(BASELINE['checkpoint']),device).eval();normalize=build_eval_transform()
    for a in arms:
        if time.time()>=DEADLINE:break
        d=run/'arms'/a['id'];d.mkdir(parents=True);(d/'probs').mkdir();result=reserve_result();save_json(d/'result.json',dict(result=str(result)))
        start=time.time()
        try:
            for row in tqdm(data_rows(),desc=a['id'],unit='image'):
                image=normalize(image=np.asarray(Image.open(row['image']).convert('RGB')))['image'].float()
                q=predict(model,image,device,**a['config']);np.save(d/'probs'/f"{row['image_id']}.npy",q);score_image(q,row,result)
                save_json(run/'progress.json',dict(arm=a['id'],image=row['image_id'],elapsed=time.time()-start))
        except torch.cuda.OutOfMemoryError as error:
            save_json(d/'failed.json',dict(reason='GPU memory cap',message=str(error),result=str(result)))
            print(a['id'],'FAILED within memory cap; preserving partial artifacts',flush=True)
            gc.collect();torch.cuda.empty_cache();continue
        s=finalize(result,a,time.time()-start,dict(cache=str(d/'probs')))
        save_json(d/'complete.json',dict(result=str(result),image_miou=s['image_miou']))
        print(a['id'],f"{s['image_miou']*100:.6f}%",flush=True);update_comparison()
    save_json(run/'status.json',dict(finished=True))


if __name__=='__main__':main()
