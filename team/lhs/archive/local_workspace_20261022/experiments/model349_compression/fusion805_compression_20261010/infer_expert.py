"""Fresh image-only Model401 probabilities using frozen 12 auxiliary parents."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
LAYOUTS = {'hr2_subset':(.5,.25,.25), 'drop_seed2_folds34':(.5,.25,.25), 'drop_seed2_folds234':(.5,.25,.25), 'drop_seed2_fold4':(.5,.25,.25), 'control':(.5,.25,.25), 'seed1':(.5,.5,0), 'seed2':(.5,0,.5), 'b4':(1,0,0), 'hr':(0,.5,.5), 'hr1':(0,1,0), 'hr2':(0,0,1)}
_parent = Path(sys.argv[sys.argv.index('--parent-package')+1]) if '--parent-package' in sys.argv else ROOT
sys.path.insert(0,str(_parent/'model397_reproduction_bundle/model329'))
sys.path.insert(0,str(_parent/'model397_reproduction_bundle/model392'))
from common import load_member, load_gray, predict_probs
from komap_seg.checkpoint import load_model_and_preprocessing, resolve_preprocessing
from komap_seg.dataset import KomapFullDataset, effective_input_scale, scaled_overlap
from komap_seg.inference import predict_probs_tta
import hashlib
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hr2-folds',type=int,nargs='+',choices=[1,2,3,4])
    p.add_argument('--layout',choices=LAYOUTS,required=True)
    p.add_argument('--parent-package',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    global ROOT
    ROOT=a.parent_package
    weights=LAYOUTS[a.layout]
    dropped={'drop_seed2_fold4':[4],'drop_seed2_folds34':[3,4],'drop_seed2_folds234':[2,3,4]}.get(a.layout,[])
    if a.layout=='hr2_subset':
        assert a.hr2_folds and len(set(a.hr2_folds))==len(a.hr2_folds)
        dropped=[f for f in [1,2,3,4] if f not in a.hr2_folds]
    else:assert a.hr2_folds is None
    seeds=[s for s in [1,2] if weights[s]]
    sys.path.insert(0,str(ROOT/'model397_reproduction_bundle/model329'))
    sys.path.insert(0,str(ROOT/'model397_reproduction_bundle/model392'))
    assert not a.output.exists()
    # args.json is bundled metadata; copy it beside the external last.pt.
    for r in json.loads((ROOT/'config/auxiliary_assets.json').read_text()):
        if r['path'].startswith('refiner/'):continue
        if any(r['path'].startswith(f'model392/seed2/fold{f}/') for f in dropped):continue
        if r['path'].startswith('model329/') and not weights[0]:continue
        if r['path'].startswith('model392/') and not any(r['path'].startswith(f'model392/seed{s}/') for s in seeds):continue
        path=a.assets/r['path'];assert sha(path)==r['sha256'],str(path)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    b4=[load_member(a.assets/f'model329/fold{f}',a.device,'last.pt') for f in range(1,5)] if weights[0] else []
    hr=[]
    for seed in seeds:
        for fold in range(1,5):
            if seed==2 and fold in dropped:continue
            model,saved=load_model_and_preprocessing(a.assets/f'model392/seed{seed}/fold{fold}/last.pt',device)
            prep=resolve_preprocessing(saved)
            assert prep['patch_size']==512 and prep['input_scale']==2.0 and prep['normalize_width'] and prep['target_width']>0
            hr.append((seed,model,prep))
    dataset=KomapFullDataset(a.input,masks_dir=None)
    a.output.mkdir(parents=True)
    rows=[]
    for i,pair in enumerate(dataset.pairs):
        sample=dataset[i];name=pair['stem'];shape=(sample['height'],sample['width'])
        # The historical archives rounded each family to FP16 before blending.
        b=predict_probs(b4,load_gray(pair['image']),tta=True,device=a.device).cpu().numpy().astype(np.float16).astype(np.float32) if b4 else np.zeros((4,*shape),dtype=np.float32)
        sums={1:np.zeros((4,*shape),dtype=np.float32),2:np.zeros((4,*shape),dtype=np.float32)}
        for seed,model,prep in hr:
            scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
            overlap=scaled_overlap(512,2.0,170,scale)
            probs=predict_probs_tta(model,sample['image'],patch_size=512,overlap=overlap,batch_size=2,tta=True,input_scale=scale,device=device,tta_mode='flip4')
            count=4-len(dropped) if seed==2 else 4
            sums[seed]+=probs.numpy().astype(np.float32)*(1./count)
        value=(weights[0]*b+sum(weights[s]*sums[s].astype(np.float16).astype(np.float32) for s in seeds))[[1,2,0,3]]
        assert np.isfinite(value).all()
        np.save(a.output/f'{name}.npy',value)
        rows.append({'name':name,'sha256':sha(a.output/f'{name}.npy')})
        print(name,flush=True)
    (a.output/'completion.json').write_text(json.dumps({'images':rows,'gt_read':False,'fresh_raw_inference':True,'layout':a.layout,'family_weights':weights,'hr_seed2_folds':[f for f in [1,2,3,4] if f not in dropped]},indent=2)+'\n')
if __name__=='__main__':main()
