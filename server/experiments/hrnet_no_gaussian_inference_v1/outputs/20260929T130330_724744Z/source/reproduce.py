"""Reconstruct a selected inference recipe from raw images, without cached predictions."""
import argparse
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from common import ROOT,BASELINE,DEADLINE,save_json,require_environment,data_rows,reserve_result,score_image,finalize,update_comparison
from inference import predict_group
from full_image import predict as full_predict
from postprocess_extra import apply as post_apply
from run_extra import ForwardAdapter
sys.path.insert(0,str(ROOT/'vendor'))
from komap_seg.checkpoint import load_model
from komap_seg.dataset import build_eval_transform


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--gpu',type=int,default=7);p.add_argument('--export-only',action='store_true');args=p.parse_args()
    require_environment();entries=json.loads((ROOT/'comparison.json').read_text());lookup={e['arm']['id']:e for e in entries}
    needed={}
    def visit(name):
        if name in needed:return
        e=lookup[name];a=e['arm']
        if 'members' in a:
            for n in a['members']:visit(n)
        elif a.get('change',{}).get('kind'):visit(a['contrast'])
        needed[name]=e
    visit(args.id)
    recipe=dict(checkpoint=BASELINE['checkpoint'],target=args.id,nodes={k:dict(arm=v['arm'],batch=v.get('batch',4),image_miou=v['image_miou']) for k,v in needed.items()})
    save_json(ROOT/'recipes'/f'{args.id}.json',recipe)
    if args.export_only:print('Exported',args.id,len(needed),'nodes');return
    torch.set_num_threads(1);torch.set_grad_enabled(False);cv2.setNumThreads(1)
    device=torch.device(f'cuda:{args.gpu}');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.36,device)
    model=load_model(Path(BASELINE['checkpoint']),device).eval();normalize=build_eval_transform()
    original_interpolation={k:v.interpolation_mode for k,v in model.decoder.blocks.items()}
    run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    name='REPRO_'+args.id;directory=run/'arms'/name;directory.mkdir(parents=True);(directory/'probs').mkdir();result=reserve_result()
    arm=dict(id=name,contrast=args.id,change={'fresh_raw_image_reproduction':True},recipe=recipe)
    save_json(run/'plan.json',[arm]);save_json(directory/'result.json',dict(result=str(result)))
    began=time.time();differences=[]
    for row in tqdm(data_rows(),desc='Fresh recipe reproduction',unit='image'):
        if time.time()>=DEADLINE:raise TimeoutError('Requested deadline reached')
        image=np.asarray(Image.open(row['image']).convert('RGB'));nodes={}
        for node,e in needed.items():
            a=e['arm'];change=a.get('change',{})
            if 'members' in a:
                weights=np.asarray(a['weights'],np.float32);weights/=weights.sum();values=[nodes[k] for k in a['members']]
                if a.get('mean','arithmetic')=='geometric':values=[np.log(np.maximum(v,1e-8)) for v in values]
                q=sum(w*v for w,v in zip(weights,values))
                if a.get('mean','arithmetic')=='geometric':q=np.exp(q-q.max(0,keepdims=True));q/=q.sum(0,keepdims=True)
            elif change.get('kind'):
                q=post_apply(nodes[a['contrast']],image,change)
            else:
                cfg=a['config'];prepared=image
                if a.get('adaptive_scale'):
                    from run_adaptive import realize
                    cfg=realize(cfg,image.shape[1],a['adaptive_scale'])
                if 'patch' in cfg and (cfg['gamma']!=1 or cfg['contrast']!=1 or cfg['sharpen']!=0):
                    x=image.astype(np.float32)/255;x=x**cfg['gamma'];x=(x-.5)*cfg['contrast']+.5
                    if cfg['sharpen']:x=x+cfg['sharpen']*(x-cv2.GaussianBlur(x,(0,0),1.))
                    prepared=np.rint(x.clip(0,1)*255).astype(np.uint8)
                inp=normalize(image=prepared)['image'].float()
                for key,block in model.decoder.blocks.items():block.interpolation_mode=original_interpolation[key]
                network=model
                if a.get('forward_transform'):
                    spec=a['forward_transform']
                    if 'decoder_interpolation' in spec:
                        for block in model.decoder.blocks.values():block.interpolation_mode=spec['decoder_interpolation']
                    else:network=ForwardAdapter(model,spec).eval()
                predict=predict_group
                if a.get('engine')=='resampling_v2':
                    from inference_resampling import predict_group as predict
                if 'patch' in cfg:q=predict(network,inp,[{**a,'config':cfg}],device,e.get('batch',4),description=node)[node]
                else:q=full_predict(network,inp,device,**cfg)
            nodes[node]=q
        probability=nodes[args.id];cached=np.load(Path(lookup[args.id]['cache'])/f"{row['image_id']}.npy")
        differences.append(dict(image_id=row['image_id'],max_abs_probability_error=float(np.abs(probability-cached).max()),
                                different_pixels=int((probability.argmax(0)!=cached.argmax(0)).sum())))
        np.save(directory/'probs'/f"{row['image_id']}.npy",probability);score_image(probability,row,result)
    s=finalize(result,arm,time.time()-began,dict(cache=str(directory/'probs'),reproduction_differences=differences))
    save_json(directory/'complete.json',dict(result=str(result),image_miou=s['image_miou']));update_comparison()
    save_json(ROOT/'checks'/f'reproduction_{args.id}.json',dict(expected=lookup[args.id]['image_miou'],observed=s['image_miou'],differences=differences,result=str(result)))
    print(name,100*s['image_miou'],'%',result,flush=True)


if __name__=='__main__':main()
