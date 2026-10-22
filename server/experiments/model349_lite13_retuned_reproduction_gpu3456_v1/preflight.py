"""Read-only checks. No optimizer, training, forward pass, or model inference."""
from pathlib import Path
import ast, importlib.metadata, os, subprocess, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ROOT, CONDA, read, sha, devices

def check():
    if Path(sys.prefix).resolve()!=CONDA.resolve():raise RuntimeError('Use conda komap through run.sh')
    for filename,digest in tqdm(read(ROOT/'code_manifest.json').items(),desc='Source hashes',mininterval=10):
        p=ROOT/filename
        if sha(p)!=digest:raise ValueError(f'Changed source: {p}')
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    for filename,digest in tqdm(read(ROOT/'data_manifest.json').items(),desc='Data hashes',mininterval=10):
        if sha(ROOT/filename)!=digest:raise ValueError(f'Changed data: {filename}')
    palette=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],dtype=np.uint8)
    counts={}
    for split,expected in [('train',70),('valid',20),('test',10)]:
        files=sorted((ROOT/f'data/{split}/images').glob('*_image.png'))
        if len(files)!=expected:raise ValueError(f'{split}: expected {expected}')
        counts[split]=len(files)
        if split=='test':continue
        if len(list((ROOT/f'data/{split}/masks').glob('*_mask.png')))!=expected:raise ValueError('Mask inventory mismatch')
        for p in tqdm(files,desc=f'{split} masks',mininterval=10):
            mask=p.parent.parent/'masks'/p.name.replace('_image','_mask')
            with Image.open(p) as im:size=im.size
            with Image.open(mask) as im:
                rgb=np.asarray(im.convert('RGB'))
                if im.size!=size:raise ValueError(f'Shape mismatch {p}')
            if not np.any(np.all(rgb[...,None,:]==palette,axis=-1),axis=-1).all():raise ValueError(f'Unknown palette {mask}')
    manifest=read(ROOT/'checks/splits/model397_split_manifest.json')
    train={p.name for p in (ROOT/'data/train/images').glob('*.png')};seen=[]
    for f,n in [(1,18),(2,18),(3,17),(4,17)]:
        base=ROOT/'checks/splits/model329_cv'
        tr={p.name for p in (base/f'train_fold{f}/images').glob('*.png')}
        va={p.name for p in (base/f'oof_fold{f}/images').glob('*.png')}
        group=lambda names:{x.split('_')[0] for x in names}
        if len(va)!=n or tr&va or tr|va!=train or group(tr)&group(va):raise ValueError('Invalid group fold')
        seen.extend(va)
    if len(seen)!=70 or set(seen)!=train:raise ValueError('OOF coverage mismatch')
    s=ROOT/'source';up=ROOT/'upstream/lite13_retuned_reproduction'
    for name in ['inference/selected_config.json','inference/selected_R_lite13.yaml','inference/selected_assets.json']:
        if (s/name).read_bytes()!=(up/name).read_bytes():raise ValueError(f'Recipe changed: {name}')
    assets=read(s/'inference/selected_assets.json')
    if len(assets['checkpoints'])!=13:raise ValueError('Expected 13 final weights')
    plan=read(ROOT/'plan.json')
    if len(plan['tasks'])!=48 or plan['training_count']!=33:raise ValueError('Unexpected training plan')
    inventory=devices();os.environ['CUDA_VISIBLE_DEVICES']=','.join(r['uuid'] for r in inventory.values())
    import torch, segmentation_models_pytorch, albumentations
    if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise RuntimeError('Expected CUDA GPUs 3-6')
    for p in ROOT.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in s.glob('run_*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    return dict(ok=True,training_started=False,inference_started=False,counts=counts,gpus=inventory,
                python=sys.executable,training_count=33,pre_prediction_stages=48,final_weights=13,
                versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','segmentation-models-pytorch','timm','albumentations','tqdm']})

if __name__=='__main__':print(check())
