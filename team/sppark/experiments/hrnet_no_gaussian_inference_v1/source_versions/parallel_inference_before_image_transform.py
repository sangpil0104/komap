"""Independent-image inference across eight GPUs; no DDP or training in this runner."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from common import ROOT,BASELINE,DEADLINE,require_environment,data_rows,reserve_result,save_json,score_image,finalize,update_comparison
from inference import DEFAULT,forward_key

STATE={}


def initialize(gpu_queue,batch,memory_fraction):
    gpu=gpu_queue.get(timeout=120)
    torch.set_num_threads(1);torch.set_grad_enabled(False);cv2.setNumThreads(1)
    device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(memory_fraction,device)
    sys.path.insert(0,str(ROOT/'vendor'))
    from komap_seg.checkpoint import load_model
    from komap_seg.dataset import build_eval_transform
    import inference
    original_tqdm=inference.tqdm
    inference.tqdm=lambda *a,**k:original_tqdm(*a,**{**k,'disable':True})
    model=load_model(Path(BASELINE['checkpoint']),device).eval()
    STATE.update(model=model,device=device,normalize=build_eval_transform(),batch=batch,gpu=gpu,
                 interpolation={k:v.interpolation_mode for k,v in model.decoder.blocks.items()})


def work(row,arms):
    from inference import predict_group
    from full_image import predict as predict_full
    from run_extra import ForwardAdapter
    from run_adaptive import realize
    if time.time()>=DEADLINE:raise TimeoutError('Deadline reached')
    pending=[a for a in arms if not (Path(a['_result'])/'per_image'/f"{row['image_id']}.json").exists()]
    if not pending:return dict(image=row['image_id'],skipped=True,gpu=STATE['gpu'])
    if pending[0].get('engine')=='resampling_v2':
        from inference_resampling import predict_group
    cfg=pending[0]['config'];image=np.asarray(Image.open(row['image']).convert('RGB'));prepared=image
    if 'patch' in cfg and (cfg['gamma']!=1 or cfg['contrast']!=1 or cfg['sharpen']!=0):
        x=image.astype(np.float32)/255;x=x**cfg['gamma'];x=(x-.5)*cfg['contrast']+.5
        if cfg['sharpen']:x=x+cfg['sharpen']*(x-cv2.GaussianBlur(x,(0,0),1.))
        prepared=np.rint(x.clip(0,1)*255).astype(np.uint8)
    tensor=STATE['normalize'](image=prepared)['image'].float()
    model=STATE['model']
    for k,b in model.decoder.blocks.items():b.interpolation_mode=STATE['interpolation'][k]
    network=model;spec=pending[0].get('forward_transform')
    if spec:
        if 'decoder_interpolation' in spec:
            for b in model.decoder.blocks.values():b.interpolation_mode=spec['decoder_interpolation']
        else:network=ForwardAdapter(model,spec).eval()
    begin=time.time();modified=copy.deepcopy(pending)
    for a in modified:
        if a.get('adaptive_scale'):
            a['config']=realize(a['config'],image.shape[1],a['adaptive_scale'])
            save_json(Path(a['_dir'])/'realized_configs'/f"{row['image_id']}.json",dict(image_width=image.shape[1],policy=a['adaptive_scale'],config=a['config']))
    if 'patch' in cfg:
        probs=predict_group(network,tensor,modified,STATE['device'],STATE['batch'],description=row['image_id'])
    else:
        assert len(modified)==1
        probs={modified[0]['id']:predict_full(network,tensor,STATE['device'],**cfg)}
    inference_seconds=time.time()-begin
    for a in pending:
        q=probs[a['id']];d=Path(a['_dir']);np.save(d/'probs'/f"{row['image_id']}.npy",q)
        score_image(q,row,a['_result'])
        save_json(d/'timing'/f"{row['image_id']}.json",dict(gpu=STATE['gpu'],inference_seconds=inference_seconds,shared_arms=[a['id'] for a in pending]))
    return dict(image=row['image_id'],gpu=STATE['gpu'],seconds=time.time()-begin,inference_seconds=inference_seconds)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--resume',type=Path);p.add_argument('--memory-fraction',type=float,default=.36);args=p.parse_args()
    require_environment();torch.set_num_threads(1);cv2.setNumThreads(1)
    gpus=[int(x) for x in args.gpus.split(',')];assert len(gpus)==len(set(gpus))
    arms=json.loads(args.plan.read_text())
    if args.resume:
        run=args.resume;assert json.loads((run/'plan.json').read_text())==arms
    else:
        run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
        save_json(run/'plan.json',arms);(run/'source').mkdir()
        for f in ROOT.glob('*.py'):(run/'source'/f.name).write_bytes(f.read_bytes())
    save_json(run/'runtime.json',dict(pid=os.getpid(),gpus=gpus,batch=args.batch,memory_fraction=args.memory_fraction,training=False))
    print('RUN:',run,flush=True)
    groups={}
    for arm in arms:
        d=run/'arms'/arm['id'];d.mkdir(parents=True,exist_ok=True)
        if (d/'complete.json').exists():continue
        if not (d/'result.json').exists():save_json(d/'result.json',dict(result=str(reserve_result())))
        arm['_dir']=str(d);arm['_result']=json.loads((d/'result.json').read_text())['result'];(d/'probs').mkdir(exist_ok=True)
        cfg=arm['config']
        if arm.get('engine')=='resampling_v2':
            from inference_resampling import forward_key as resample_key
            key=resample_key(cfg)
        else:key=forward_key(cfg) if 'patch' in cfg else ('full',arm['id'])
        key=(arm.get('engine'),key,json.dumps(arm.get('forward_transform')),json.dumps(arm.get('adaptive_scale')))
        groups.setdefault(key,[]).append(arm)
    ctx=mp.get_context('spawn');gpu_queue=ctx.Queue()
    for gpu in gpus:gpu_queue.put(gpu)
    completed=[];failed=[];start=time.time()
    with ProcessPoolExecutor(max_workers=len(gpus),mp_context=ctx,initializer=initialize,initargs=(gpu_queue,args.batch,args.memory_fraction)) as pool:
        for group in groups.values():
            if time.time()>=DEADLINE-60:break
            begin=time.time();futures=[pool.submit(work,row,group) for row in data_rows()];errors=[]
            with tqdm(total=20,desc=group[0]['id']+' valid20 / '+str(len(gpus))+' GPUs',unit='image') as bar:
                for f in as_completed(futures):
                    try:
                        record=f.result();save_json(run/'progress.json',dict(group=group[0]['id'],elapsed=time.time()-start,**record))
                    except Exception as error:
                        errors.append(repr(error));print('Image inference failed:',repr(error),flush=True)
                    bar.update()
            if errors:
                for a in group:save_json(Path(a['_dir'])/'failed.json',dict(errors=errors,result=a['_result']));failed.append(a['id'])
                continue
            for a in group:
                public={k:v for k,v in a.items() if not k.startswith('_')}
                s=finalize(a['_result'],public,time.time()-begin,dict(gpus=gpus,batch=args.batch,cache=str(Path(a['_dir'])/'probs'),execution='independent images across GPUs'))
                save_json(Path(a['_dir'])/'complete.json',dict(result=a['_result'],image_miou=s['image_miou']));completed.append(a['id'])
                print(a['id'],f"image mIoU={100*s['image_miou']:.6f}%",a['_result'],flush=True)
            update_comparison()
    save_json(run/'status.json',dict(elapsed=time.time()-start,completed=completed,failed=failed))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
