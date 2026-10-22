"""CPU-only immutable HR336 BatchNorm inventory; no image or labels read."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'Model349_exh14'
sys.path.insert(0,str(SOURCE/'src'))
from komap_seg.checkpoint import load_model
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(2);checkpoint=SOURCE/'weights/model336_hrnet_best/last.pt';before=sha(checkpoint)
 frozen=json.loads((ROOT.parent/'accuracy81_residual_20261008/parent_frozen.json').read_text())
 assert before==frozen['weights/model336_hrnet_best/last.pt']
 model=load_model(checkpoint,torch.device('cpu'));rows=[]
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
   assert not module.training and module.track_running_stats and module.running_mean is not None and module.running_var is not None
   assert torch.isfinite(module.running_mean).all() and torch.isfinite(module.running_var).all() and (module.running_var>=0).all()
   rows.append({'name':name,'type':type(module).__name__,'channels':module.num_features,'affine':module.affine,'eps':module.eps,'minimum_running_variance':float(module.running_var.min())})
 assert rows and sha(checkpoint)==before
 result={'checkpoint_sha256':before,'model_loaded_strictly_from_original_checkpoint':True,'bn_layers':rows,'bn_layer_count':len(rows),'total_bn_channels':sum(x['channels'] for x in rows),'gpu_used':False,'any_image_or_gt_opened':False,'checkpoint_unchanged':True}
 (ROOT/'bn_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['bn_layer_count','total_bn_channels','gpu_used','any_image_or_gt_opened','checkpoint_unchanged']}))
if __name__=='__main__':main()
