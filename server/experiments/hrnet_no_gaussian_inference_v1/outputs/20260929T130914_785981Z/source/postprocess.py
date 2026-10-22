"""CPU-only, label-independent inference postprocessing of cached probabilities."""
import argparse
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from common import ROOT,DEADLINE,data_rows,save_json,reserve_result,score_image,finalize,update_comparison,require_environment


def normalize(p):
    p=np.maximum(p,0)
    return p/np.maximum(p.sum(0,keepdims=True),1e-10)


def guided(p,gray,radius,eps):
    # Scalar grayscale guide; per-class local linear fit, no target-mask input.
    k=(2*radius+1,)*2
    mean=lambda x:cv2.boxFilter(x,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT_101)
    mI=mean(gray);var=mean(gray*gray)-mI*mI
    result=[]
    for pc in p:
        mp=mean(pc);a=(mean(gray*pc)-mI*mp)/(var+eps);b=mp-a*mI
        result.append(mean(a)*gray+mean(b))
    return normalize(np.stack(result))


def apply(p,image,change):
    kind=change['kind']
    if kind=='class_weight':
        q=p.copy();q[change['class']]*=change['weight'];return normalize(q)
    if kind in ('smooth','sharpen'):
        smooth=np.stack([cv2.GaussianBlur(x,(0,0),change.get('sigma',1.)) for x in p])
        return normalize(smooth if kind=='smooth' else p+change['amount']*(p-smooth))
    if kind=='guided':
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        return guided(p,gray,change['radius'],change['eps'])
    if kind=='components':
        pred=p.argmax(0);q=p.copy()
        for cls in change.get('classes',[0,1,2,3]):
            n,lab,stats,_=cv2.connectedComponentsWithStats((pred==cls).astype(np.uint8),connectivity=8)
            sizes=stats[:,cv2.CC_STAT_AREA];small=sizes<change['area'];small[0]=False
            q[cls,small[lab]]=0
        return normalize(q)
    if kind=='median_confidence':
        pred=p.argmax(0);filtered=cv2.medianBlur(pred.astype(np.uint8),3)
        sortedp=np.partition(p,-2,axis=0);uncertain=sortedp[-1]-sortedp[-2]<change['margin']
        # Median of IDs would impose arbitrary class ordering: use per-class votes instead.
        votes=np.stack([cv2.boxFilter((pred==k).astype(np.float32),-1,(3,3)) for k in range(4)])
        majority=votes.argmax(0);q=p.copy();ys,xs=np.where(uncertain & (majority!=pred))
        q[majority[ys,xs],ys,xs]=1.01
        return normalize(q)
    raise ValueError(kind)


def plan():
    changes=[]
    for cls in (0,1,2,3):
        for weight in (.9,.95,1.05,1.1,1.2):changes.append(dict(kind='class_weight',**{'class':cls},weight=weight))
    for sigma in (.35,.5,.75,1.):changes.append(dict(kind='smooth',sigma=sigma))
    for amount in (.25,.5,1.):changes.append(dict(kind='sharpen',amount=amount,sigma=1.))
    for radius in (1,2,4):
        for eps in (.0004,.0025,.01):changes.append(dict(kind='guided',radius=radius,eps=eps))
    for area in (2,4,8,16):changes.append(dict(kind='components',area=area))
    for margin in (.05,.1,.2):changes.append(dict(kind='median_confidence',margin=margin))
    return changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--control',default='A00_control')
    p.add_argument('--prefix',default='P');p.add_argument('--plan',type=Path);args=p.parse_args()
    require_environment();cv2.setNumThreads(1)
    cache=args.source/'arms'/args.control/'probs';assert len(list(cache.glob('*.npy')))==20
    run=ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    changes=json.loads(args.plan.read_text()) if args.plan else plan()
    arms=[dict(id=f'{args.prefix}{i:02d}_{c["kind"]}',contrast=args.control,change=c,cache_source=str(cache)) for i,c in enumerate(changes)]
    save_json(run/'plan.json',arms)
    print('RUN:',run,flush=True)
    rows=data_rows();started=time.time()
    for a in tqdm(arms,desc='CPU postprocess valid20',unit='condition',mininterval=2):
        if time.time()>=DEADLINE:break
        d=run/'arms'/a['id'];d.mkdir(parents=True);result=reserve_result();save_json(d/'result.json',dict(result=str(result)))
        (d/'probs').mkdir()
        begin=time.time()
        for row in rows:
            if time.time()>=DEADLINE:raise TimeoutError('Requested deadline reached')
            prob=np.load(cache/f"{row['image_id']}.npy");image=np.asarray(Image.open(row['image']).convert('RGB'))
            q=apply(prob,image,a['change']);np.save(d/'probs'/f"{row['image_id']}.npy",q)
            score_image(q,row,result)
        summary=finalize(result,a,time.time()-begin,dict(cache=str(d/'probs')))
        save_json(d/'complete.json',dict(result=str(result),image_miou=summary['image_miou']))
        print(f"{a['id']}: {100*summary['image_miou']:.6f}% ({a['change']})",flush=True)
        update_comparison()
        save_json(run/'progress.json',dict(completed=a['id'],elapsed=time.time()-started))
    save_json(run/'status.json',dict(elapsed=time.time()-started))


if __name__=='__main__':main()
