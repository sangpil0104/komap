"""Image-only HR seed2 fold maps; control parity with frozen family cache."""
from pathlib import Path
import os,sys,json,time,argparse
import numpy as np
import torch
from PIL import Image
import hashlib
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
ROOT=Path(__file__).resolve().parent
PARENT=Path('/workspace/model349_compress/accuracy81_stage1_fresh_20261009')
sys.path.insert(0,str(PARENT/'model397_reproduction_bundle/model392'))
from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
from komap_seg.inference import predict_probs_tta

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();assert a.shard in [0,1]
 out=ROOT/f'fold_maps/shard{a.shard}';out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda')
 manifest=json.loads((PARENT/'config/auxiliary_assets.json').read_text());expected={r['path']:r['sha256'] for r in manifest};models=[];frozen={}
 for fold in range(1,5):
  key=f'model392/seed2/fold{fold}/last.pt';path=PARENT/'assets'/key;frozen[str(path)]=sha(path);assert frozen[str(path)]==expected[key]
  model,saved=load_model_and_preprocessing(path,device);prep=resolve_preprocessing(saved);models.append((fold,model,prep))
 dataset=KomapFullDataset('/workspace/komap_model1/data/valid/images',masks_dir=None);rows=[];start=time.time()
 for i,pair in enumerate(dataset.pairs):
  if i%2!=a.shard:continue
  sample=dataset[i];name=pair['stem'];shape=(sample['height'],sample['width']);parts=[]
  for fold,model,prep in models:
   scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width']);overlap=scaled_overlap(512,2.,170,scale)
   probs=predict_probs_tta(model,sample['image'],patch_size=512,overlap=overlap,batch_size=2,tta=True,input_scale=scale,device=device,tta_mode='flip4').numpy().astype(np.float32)
   assert probs.shape==(4,*shape) and np.isfinite(probs).all();np.save(out/f'{name}_fold{fold}.npy',probs);parts.append(probs)
  mean=sum(x*.25 for x in parts).astype(np.float16)
  with np.load(f'/workspace/komap_model1/outputs/model401_model397_blend_valid_once/hrnet_seed2_mean_valid/{name}.npz',allow_pickle=False) as z:cached=z[name]
  different=int(np.count_nonzero(mean!=cached));error=float(np.max(np.abs(mean.astype(np.float32)-cached.astype(np.float32))))
  rows.append({'name':name,'family_changed_values':different,'max_abs_difference':error});print(name,round(time.time()-start,1),different,error,flush=True)
 assert len(rows)==10 and all(sha(p)==h for p,h in frozen.items())
 (out/'completion.json').write_text(json.dumps({'rows':rows,'gt_read':False,'checkpoint_sha256':frozen,'elapsed_seconds':time.time()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']},indent=2)+'\n')
if __name__=='__main__':
 import traceback
 try:main()
 except BaseException:(ROOT/f'fold_failure_{sys.argv[-1]}.txt').write_text(traceback.format_exc());raise
