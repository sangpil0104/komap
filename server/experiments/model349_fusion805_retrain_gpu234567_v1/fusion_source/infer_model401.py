"""Fresh image-only Model401 probabilities using frozen 12 auxiliary parents."""
from pathlib import Path
import argparse, json, sys
from tqdm import tqdm
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model397_reproduction_bundle/model329'))
sys.path.insert(0,str(ROOT/'model397_reproduction_bundle/model392'))
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
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    assert not a.output.exists()
    # args.json is bundled metadata; copy it beside the external last.pt.
    for r in json.loads((ROOT/'config/auxiliary_assets.json').read_text()):
        if r['path'].startswith('refiner/'):continue
        path=a.assets/r['path'];assert sha(path)==r['sha256'],str(path)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    b4=[load_member(a.assets/f'model329/fold{f}',a.device,'last.pt') for f in range(1,5)]
    hr=[]
    for seed in [1,2]:
        for fold in range(1,5):
            model,saved=load_model_and_preprocessing(a.assets/f'model392/seed{seed}/fold{fold}/last.pt',device)
            prep=resolve_preprocessing(saved)
            assert prep['patch_size']==512 and prep['input_scale']==2.0 and prep['normalize_width'] and prep['target_width']>0
            hr.append((seed,model,prep))
    dataset=KomapFullDataset(a.input,masks_dir=None)
    a.output.mkdir(parents=True)
    rows=[]
    for i,pair in enumerate(tqdm(dataset.pairs,desc="Expert valid images",unit="image")):
        sample=dataset[i];name=pair['stem'];shape=(sample['height'],sample['width'])
        # The historical archives rounded each family to FP16 before blending.
        b=predict_probs(b4,load_gray(pair['image']),tta=True,device=a.device).cpu().numpy().astype(np.float16).astype(np.float32)
        sums={1:np.zeros((4,*shape),dtype=np.float32),2:np.zeros((4,*shape),dtype=np.float32)}
        for seed,model,prep in hr:
            scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
            overlap=scaled_overlap(512,2.0,170,scale)
            probs=predict_probs_tta(model,sample['image'],patch_size=512,overlap=overlap,batch_size=2,tta=True,input_scale=scale,device=device,tta_mode='flip4')
            sums[seed]+=probs.numpy().astype(np.float32)*.25
        value=(.5*b+.25*sums[1].astype(np.float16).astype(np.float32)+.25*sums[2].astype(np.float16).astype(np.float32))[[1,2,0,3]]
        assert np.isfinite(value).all()
        np.save(a.output/f'{name}.npy',value)
        rows.append({'name':name,'sha256':sha(a.output/f'{name}.npy')})
        print(name,flush=True)
    (a.output/'completion.json').write_text(json.dumps({'images':rows,'gt_read':False,'fresh_raw_inference':True},indent=2)+'\n')
if __name__=='__main__':main()
