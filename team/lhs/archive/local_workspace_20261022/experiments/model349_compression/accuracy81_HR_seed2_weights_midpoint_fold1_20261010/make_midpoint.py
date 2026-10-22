import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import torch,json,hashlib
R=Path(__file__).resolve().parent
P=Path('/workspace/komap_model1/outputs/model392_widthnorm/seed2/fold1/last.pt')
Q=R.parent/'accuracy81_HR_seed2_warmstart_fold1_20261010/original/last.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'midpoint.pt').exists()
h={str(p):sha(p) for p in [P,Q]}
a=torch.load(P,map_location='cpu',weights_only=False);b=torch.load(Q,map_location='cpu',weights_only=False)
assert a['model_config']==b['model_config']
pa=dict(a['preprocessing_config']);pb=dict(b['preprocessing_config'])
assert pa.pop('input_mode','original')==pb.pop('input_mode','original')=='original'
assert pa==pb
x=a['state_dict'];y=b['state_dict'];assert x.keys()==y.keys()
changed=[];buffers=[]
for k,v in x.items():
 w=y[k];assert v.shape==w.shape and v.dtype==w.dtype and torch.isfinite(v).all() and torch.isfinite(w).all()
 if k.endswith(('running_mean','running_var','num_batches_tracked')) or not v.is_floating_point():
  assert torch.equal(v,w),k;buffers.append(k)
 else:
  if not torch.equal(v,w):changed.append(k)
  x[k]=v*.5+w*.5
assert changed
assert all(sha(Path(p))==v for p,v in h.items())
torch.save(a,R/'midpoint.pt')
(R/'midpoint_receipt.json').write_text(json.dumps({'inputs_unchanged':True,'input_sha256':h,'changed_tensors':len(changed),'identical_buffers':len(buffers),'output_sha256':sha(R/'midpoint.pt')},indent=2))
