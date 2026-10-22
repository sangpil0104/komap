"""Recompute an inference recipe from raw images on eight GPUs, then compare caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from common import ROOT, BASELINE, DEADLINE, require_environment, data_rows, reserve_result, save_json, score_image, finalize, update_comparison
import parallel_inference as parallel
from inference import predict_group
from full_image import predict as full_predict
from postprocess_extra import apply as post_apply
from run_extra import ForwardAdapter


def dependencies(target, lookup):
    nodes={}
    def visit(name):
        if name in nodes:return
        entry=lookup[name];arm=entry['arm']
        if 'members' in arm:
            for member in arm['members']:visit(member)
        elif arm.get('change',{}).get('kind'):visit(arm['contrast'])
        nodes[name]=dict(arm=arm,batch=entry.get('batch',4))
    visit(target)
    return nodes


@torch.inference_mode()
def reconstruct(image,recipe):
    state=parallel.STATE;model=state['model'];device=state['device'];nodes={}
    for name,entry in recipe['nodes'].items():
        if time.time()>=DEADLINE:raise TimeoutError('Requested deadline reached')
        arm=entry['arm'];change=arm.get('change',{})
        if 'members' in arm:
            weights=np.asarray(arm['weights'],np.float32);weights/=weights.sum();q=None
            for member,weight in zip(arm['members'],weights):
                value=nodes[member]
                if arm.get('mean','arithmetic')=='geometric':value=np.log(np.maximum(value,1e-8))
                q=weight*value if q is None else q+weight*value
            if arm.get('mean','arithmetic')=='geometric':q=np.exp(q-q.max(0,keepdims=True));q/=q.sum(0,keepdims=True)
        elif change.get('kind'):
            q=post_apply(nodes[arm['contrast']],image,change)
        else:
            cfg=arm['config'];prepared=image
            if arm.get('adaptive_scale'):
                from run_adaptive import realize
                cfg=realize(cfg,image.shape[1],arm['adaptive_scale'])
            if 'patch' in cfg and (cfg['gamma']!=1 or cfg['contrast']!=1 or cfg['sharpen']!=0):
                x=image.astype(np.float32)/255;x=x**cfg['gamma'];x=(x-.5)*cfg['contrast']+.5
                if cfg['sharpen']:x=x+cfg['sharpen']*(x-cv2.GaussianBlur(x,(0,0),1.))
                prepared=np.rint(x.clip(0,1)*255).astype(np.uint8)
            if arm.get('input_transform'):
                from image_transform import apply
                prepared=apply(prepared,arm['input_transform'])
            tensor=state['normalize'](image=prepared)['image'].float()
            for key,block in model.decoder.blocks.items():block.interpolation_mode=state['interpolation'][key]
            network=model;spec=arm.get('forward_transform')
            if spec:
                if 'decoder_interpolation' in spec:
                    for block in model.decoder.blocks.values():block.interpolation_mode=spec['decoder_interpolation']
                else:network=ForwardAdapter(model,spec).eval()
            if 'patch' in cfg:
                predict=predict_group
                if arm.get('engine')=='resampling_v2':
                    from inference_resampling import predict_group as predict
                q=predict(network,tensor,[{**arm,'config':cfg}],device,entry['batch'],description=name)[name]
            else:
                predict=full_predict
                if arm.get('engine')=='full_v2':
                    from full_image_extended import predict
                q=predict(network,tensor,device,**cfg)
        nodes[name]=q
    return nodes[recipe['target']]


def work(row,recipe,directory,result,reference):
    image=np.asarray(Image.open(row['image']).convert('RGB'));begin=time.time()
    q=reconstruct(image,recipe)
    # Reference probabilities enter only after raw-image inference has completed.
    cached=np.load(Path(reference)/f"{row['image_id']}.npy")
    difference=dict(image_id=row['image_id'],max_abs_probability_error=float(np.abs(q-cached).max()),
                    different_pixels=int((q.argmax(0)!=cached.argmax(0)).sum()),inference_seconds=time.time()-begin)
    np.save(Path(directory)/'probs'/f"{row['image_id']}.npy",q);score_image(q,row,result)
    return difference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',required=True);parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    args=parser.parse_args();require_environment()
    lookup={e['arm']['id']:e for e in json.loads((ROOT/'comparison.json').read_text())}
    nodes=dependencies(args.id,lookup)
    recipe=dict(checkpoint=BASELINE['checkpoint'],target=args.id,nodes=nodes)
    save_json(ROOT/'recipes'/f'{args.id}.parallel.json',recipe)
    run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    name='REPRO8_'+args.id;directory=run/'arms'/name;directory.mkdir(parents=True);(directory/'probs').mkdir()
    (run/'source').mkdir()
    for file in ROOT.glob('*.py'):(run/'source'/file.name).write_bytes(file.read_bytes())
    result=reserve_result();save_json(directory/'result.json',dict(result=str(result)))
    arm=dict(id=name,contrast=args.id,change={'fresh_raw_image_reproduction':True},recipe=recipe)
    save_json(run/'plan.json',[arm]);print('RUN:',run,flush=True)
    gpus=[int(gpu) for gpu in args.gpus.split(',')];assert len(gpus)==len(set(gpus))
    ctx=mp.get_context('spawn');gpu_queue=ctx.Queue()
    for gpu in gpus:gpu_queue.put(gpu)
    began=time.time();differences=[]
    with ProcessPoolExecutor(max_workers=len(gpus),mp_context=ctx,initializer=parallel.initialize,initargs=(gpu_queue,4,.36)) as pool:
        jobs=[pool.submit(work,row,recipe,str(directory),str(result),lookup[args.id]['cache']) for row in data_rows()]
        for job in tqdm(as_completed(jobs),total=len(jobs),desc='Fresh raw-image reproduction / 8 GPUs',unit='image'):
            differences.append(job.result())
    differences.sort(key=lambda r:r['image_id'])
    summary=finalize(result,arm,time.time()-began,dict(cache=str(directory/'probs'),reproduction_differences=differences,gpus=gpus))
    save_json(directory/'complete.json',dict(result=str(result),image_miou=summary['image_miou']));update_comparison()
    record=dict(expected=lookup[args.id]['image_miou'],observed=summary['image_miou'],differences=differences,result=str(result),
                all_probabilities_bitwise_equal=all(r['max_abs_probability_error']==0 for r in differences))
    save_json(ROOT/'checks'/f'reproduction8_{args.id}.json',record)
    print(name,100*summary['image_miou'],'%',record['all_probabilities_bitwise_equal'],flush=True)
    if any(r['different_pixels'] for r in differences):raise RuntimeError('Fresh reconstruction changed predicted pixels; inspect saved differences')


if __name__=='__main__':main()
