#!/usr/bin/env python
"""Evaluate frozen HRNet weights, never train. GPU memory is explicitly bounded."""
import argparse
import json
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import ROOT,BASELINE,DEADLINE,save_json,require_environment,data_rows,reserve_result,score_image,finalize,update_comparison,sha
from inference import DEFAULT,predict_group,forward_key

sys.path.insert(0,str(ROOT/'vendor'))
from komap_seg.checkpoint import load_model
from komap_seg.dataset import build_eval_transform


def first_plan():
    changes=[('A00_control',{}),('A01_area',{'restore':'area'}),('A02_bicubic',{'restore':'bicubic'}),
             ('A03_restore_aa',{'restore_antialias':True}),('A04_logits_restore',{'restore_logits':True}),
             ('A05_gaussian_025',{'sigma':.25}),('A06_uniform',{'window':'uniform'}),
             ('A07_hann',{'window':'hann'}),('A08_gaussian_01875',{'sigma':.1875}),
             ('A09_gaussian_00625',{'sigma':.0625}),('A10_no_tta',{'tta':'none'}),
             ('A11_hflip',{'tta':'hflip'}),('A12_vflip',{'tta':'vflip'}),
             ('A13_logit_tiles',{'tile_merge':'logit'}),('A14_log_tta',{'tta_merge':'log'})]
    return [dict(id=name,contrast='A00_control' if change else 'original_saved_masks',change=change,
                 config={**DEFAULT,**change}) for name,change in changes]


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--plan',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--limit',type=int,default=20)
    p.add_argument('--memory-fraction',type=float,default=.36);args=p.parse_args()
    require_environment();torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device=torch.device(f'cuda:{args.gpu}');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    if time.time()>=DEADLINE:raise SystemExit('Deadline reached')
    arms=json.loads(args.plan.read_text()) if args.plan else first_plan()
    if args.resume:
        run=args.resume;assert json.loads((run/'plan.json').read_text())==arms
    else:
        run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
        save_json(run/'plan.json',arms);save_json(run/'runtime.json',dict(gpu=args.gpu,batch=args.batch,pid=os.getpid(),memory_fraction=args.memory_fraction,deadline=DEADLINE))
        (run/'source').mkdir()
        for source in ROOT.glob('*.py'):(run/'source'/source.name).write_bytes(source.read_bytes())
    print('RUN:',run,flush=True)
    groups={}
    for arm in arms:
        cfg=arm['config'];assert set(cfg)==set(DEFAULT)
        d=run/'arms'/arm['id'];d.mkdir(parents=True,exist_ok=True)
        if (d/'complete.json').exists():continue
        if not (d/'result.json').exists():save_json(d/'result.json',{'result':str(reserve_result())})
        arm['_dir']=str(d);arm['_result']=json.loads((d/'result.json').read_text())['result']
        (d/'probs').mkdir(exist_ok=True)
        groups.setdefault(forward_key(cfg),[]).append(arm)
    if not groups:update_comparison();return
    model=load_model(Path(BASELINE['checkpoint']),device)
    model.eval();transform=build_eval_transform();rows=data_rows()[:args.limit]
    started=time.time()
    for group in groups.values():
        gs=time.time()
        for row in tqdm(rows,desc=group[0]['id']+' valid20',unit='image',mininterval=2):
            pending=[a for a in group if not (Path(a['_result'])/'per_image'/f"{row['image_id']}.json").exists()]
            if not pending:continue
            image=np.asarray(Image.open(row['image']).convert('RGB'))
            cfg=group[0]['config']
            if cfg['gamma']!=1 or cfg['contrast']!=1 or cfg['sharpen']!=0:
                import cv2
                x=image.astype(np.float32)/255
                x=x**cfg['gamma'];x=(x-.5)*cfg['contrast']+.5
                if cfg['sharpen']:x=x+cfg['sharpen']*(x-cv2.GaussianBlur(x,(0,0),1.))
                image=np.rint(x.clip(0,1)*255).astype(np.uint8)
            inp=transform(image=image)['image'].float()
            probs=predict_group(model,inp,pending,device,args.batch,description=row['image_id'])
            for a in pending:
                name=a['id'];directory=Path(a['_dir']);np.save(directory/'probs'/f"{row['image_id']}.npy",probs[name])
                score_image(probs[name],row,a['_result'])
            save_json(run/'progress.json',dict(image=row['image_id'],elapsed=time.time()-started,group=group[0]['id'],time_remaining=max(0,DEADLINE-time.time()),pid=os.getpid()))
        if args.limit==20:
            for a in group:
                public={k:v for k,v in a.items() if not k.startswith('_')}
                summary=finalize(a['_result'],public,time.time()-gs,dict(gpu=args.gpu,batch=args.batch,cache=str(Path(a['_dir'])/'probs')))
                save_json(Path(a['_dir'])/'complete.json',dict(result=a['_result'],image_miou=summary['image_miou']))
                print(f"{a['id']}: image mIoU={100*summary['image_miou']:.6f}% -> {a['_result']}",flush=True)
            update_comparison()
    save_json(run/'status.json',dict(complete=args.limit==20,elapsed=time.time()-started,peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20))


if __name__=='__main__':main()
