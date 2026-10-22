"""Train-only held-out expert features for the bounded residual model F."""
import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from experiment import ROOT,read_json,write_json,sha

PALETTE=np.asarray([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)


def blend(b4,hr1,hr2):
    arrays=[np.asarray(v,dtype=np.float32) for v in [b4,hr1,hr2]]
    if len({a.shape for a in arrays})!=1 or arrays[0].shape[0]!=4:raise ValueError('Expert shape mismatch.')
    result=(.5*arrays[0]+.25*arrays[1]+.25*arrays[2])[[1,2,0,3]]
    if not np.isfinite(result).all() or result.min() < -1e-4 or result.max()>1.0001 or np.max(np.abs(result.sum(0)-1))>.03:
        raise ValueError('Invalid expert probability values.')
    return result


def decode(path):
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
    target=np.full(rgb.shape[:2],255,np.uint8)
    for c,color in enumerate(PALETTE):target[np.all(rgb==color,axis=-1)]=c
    if np.any(target==255):raise ValueError(f'Unknown mask colors: {path}')
    return target


def prepare(run,data_root,splits,output):
    output.mkdir(parents=True,exist_ok=False)
    records=[];seen=set()
    for fold in tqdm(range(1,5),desc='Build train70 OOF features',unit='fold'):
        held={p.stem.removesuffix('_image') for p in (splits/f'model329_cv/oof_fold{fold}/images').glob('*.png')}
        train={p.stem.removesuffix('_image') for p in (splits/f'model329_cv/train_fold{fold}/images').glob('*.png')}
        group=lambda names:{re.match(r'A\d+',n).group() for n in names}
        if held&train or group(held)&group(train) or held&seen:raise ValueError('OOF fold leakage/duplication.')
        paths=[run/f'expert_oof/{family}/fold{fold}/probabilities.npz' for family in ['b4','hr1','hr2']]
        arrays=[np.load(p,allow_pickle=False) for p in paths]
        try:
            if any(set(a.files)!=held for a in arrays):raise ValueError('OOF archive image set mismatch.')
            hashes={str(p):sha(p) for p in paths}
            for name in sorted(held):
                probability=blend(*(a[name] for a in arrays))
                image_path=data_root/f'train/images/{name}_image.png'
                mask_path=data_root/f'train/masks/{name}_mask.png'
                with Image.open(image_path) as im:gray=np.asarray(im.convert('L'),np.float32)/255.
                target=decode(mask_path)
                if probability.shape!=(4,*target.shape) or gray.shape!=target.shape:raise ValueError('Feature shape mismatch.')
                path=output/f'{name}.npz'
                np.savez_compressed(path,logits=np.log(np.maximum(probability,1e-7)).astype(np.float32),gray=gray,target=target)
                records.append(dict(name=name,fold=fold,features_sha256=sha(path),image_sha256=sha(image_path),
                    train_mask_sha256=sha(mask_path),oof_archives=hashes))
            seen|=held
        finally:
            for a in arrays:a.close()
    expected={p.stem.removesuffix('_image') for p in (data_root/'train/images').glob('*.png')}
    if len(records)!=70 or seen!=expected:raise ValueError('Require exact train70 coverage.')
    write_json(output/'manifest.json',dict(split='train',images=records,valid_read=False,test_read=False,
        source='New Model397 group OOF: B4 0.50 + HR seed1 0.25 + HR seed2 0.25',
        class_order=['Al3Ni','Primary Si','Al','Eutectic Si'],feature='log(max(probability,1e-7))',
        gray='raw L/255; per-image zscore applied by the unmodified gray_input.prepare_gray'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare(a.run,a.data_root,a.splits,a.output)


if __name__=='__main__':main()
