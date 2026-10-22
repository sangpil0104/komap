import gc,json,os,sys
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'model397/source/model392'))
from komap_seg.model import build_model
report=[]
for gpu in [4,5,6,7]:
 device=torch.device(f'cuda:{gpu}')
 for family in ['b4','hrnet']:
  model=(smp.Unet(encoder_name='efficientnet-b4',encoder_weights='imagenet',in_channels=3,classes=4) if family=='b4' else build_model()).eval().to(device)
  size,bs=(1536,2) if family=='b4' else (512,4)
  torch.cuda.reset_peak_memory_stats(device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   y=model(torch.zeros(bs,3,size,size,device=device))
  assert y.shape==(bs,4,size,size) and torch.isfinite(y).all()
  row=dict(gpu=gpu,family=family,input=[bs,3,size,size],output=list(y.shape),forward_peak_gib=torch.cuda.max_memory_allocated(device)/2**30,training_started=False)
  print(json.dumps(row),flush=True); report.append(row)
  del model,y; gc.collect(); torch.cuda.empty_cache()
(root/'reports/model_forward.json').write_text(json.dumps(report,indent=2))
