"""Image-only expert family inference with the upstream numerical ordering."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--family',choices=['b4','hr1','hr2'],required=True)
    p.add_argument('--fold',type=int,choices=[1,2,3,4])
    p.add_argument('--fold-ids',default='1,2,3,4')
    p.add_argument('--b4-dir',choices=['model329','model329_ws'],default='model329')
    p.add_argument('--split',choices=['valid','test'],default='valid')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    import cv2
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if torch.cuda.device_count()!=1:raise RuntimeError('Each expert worker must see one assigned GPU.')
    sys.path[:0]=[str(a.bundle/'model397_reproduction_bundle/model329'),str(a.bundle/'model397_reproduction_bundle/model392')]
    from common import load_member,load_gray,predict_probs
    from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
    from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
    from komap_seg.inference import predict_probs_tta
    directory=a.data_root/f'{a.split}/images' if a.fold is None else a.splits/f'model329_cv/oof_fold{a.fold}/images'
    dataset=KomapFullDataset(directory,masks_dir=None)
    folds=[a.fold] if a.fold else [int(x) for x in a.fold_ids.split(',')]
    if not folds or len(folds)!=len(set(folds)) or not set(folds)<=set(range(1,5)):
        raise ValueError('Invalid fold subset')
    if a.fold and a.b4_dir!='model329':raise ValueError('F OOF requires original B4 weights')
    device=torch.device('cuda:0');models=[];checkpoints=[]
    for fold in folds:
        checkpoint=a.run/'expert'/(a.b4_dir if a.family=='b4' else f'model392/seed{a.family[-1]}')/f'fold{fold}/last.pt'
        checkpoints.append(str(checkpoint))
        if a.family=='b4':models.append(load_member(checkpoint.parent,'cuda:0','last.pt'))
        else:
            model,saved=load_model_and_preprocessing(checkpoint,device)
            prep=resolve_preprocessing(saved)
            if not (prep['patch_size']==512 and prep['input_scale']==2.0 and prep['normalize_width'] and prep['target_width']>0):
                raise ValueError(f'Unexpected expert preprocessing: {prep}')
            models.append((model,prep))
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();values={}
    for index,pair in enumerate(tqdm(dataset.pairs,desc=f'{a.family} '+(a.split if a.fold is None else f'OOF fold{a.fold}'),unit='image')):
        sample=dataset[index];shape=(sample['height'],sample['width'])
        if a.family=='b4':
            value=predict_probs(models,load_gray(pair['image']),tta=True,device='cuda:0').cpu().numpy()
        else:
            value=np.zeros((4,*shape),np.float32)
            for model,prep in models:
                scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
                overlap=scaled_overlap(512,2.0,170,scale)
                probs=predict_probs_tta(model,sample['image'],patch_size=512,overlap=overlap,
                    batch_size=2,tta=True,input_scale=scale,device=device,tta_mode='flip4')
                value+=probs.numpy().astype(np.float32)*(1.0/len(models))
        if value.shape!=(4,*shape) or not np.isfinite(value).all():raise ValueError('Invalid expert probabilities.')
        # Round only the family average, not each fold before averaging.
        values[pair['stem']]=value.astype(np.float16)
    np.savez_compressed(a.output/'probabilities.npz',**values)
    (a.output/'completion.json').write_text(json.dumps(dict(family=a.family,fold=a.fold,fold_ids=folds,b4_dir=a.b4_dir,images=sorted(values),
        checkpoint_paths=checkpoints,gt_read=False,family_rounding='float16 after fold mean',
        elapsed_seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()
