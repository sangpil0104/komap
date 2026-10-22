"""Official hard-IoU equivalence including absent classes, CPU-only synthetic data."""
import torch,numpy as np
from pathlib import Path
from train_fullimage import soft_iou,objective
from residual_model import Residual
from fields_io import sha,write
ROOT=Path(__file__).resolve().parent;torch.manual_seed(81349);torch.use_deterministic_algorithms(True)
target=torch.randint(4,(2,24,32));target[1]=2
pred=torch.randint(4,(2,24,32));pred[1]=2
hard=torch.nn.functional.one_hot(pred,4).permute(0,3,1,2).float();actual=soft_iou(hard,target).numpy();expected=[]
for i in range(2):
 row=[]
 for c in range(4):
  a=pred[i].numpy()==c;b=target[i].numpy()==c;u=int(np.count_nonzero(a|b));row.append(float(np.count_nonzero(a&b)/u) if u else 0.)
 expected.append(row)
assert np.allclose(actual,expected,rtol=0,atol=1e-7) and np.array_equal(actual[1],[0,0,1,0])
model=Residual();logits=torch.randn(1,4,32,32);gray=torch.rand(1,1,32,32);gt=torch.randint(4,(1,32,32))
value,delta=model(logits,gray);assert torch.equal(value,logits) and torch.count_nonzero(delta)==0
loss=objective(value,gt,delta);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
write(ROOT/'smoke_fullimage.json',{'official_hard_iou_agrees_including_empty_class':True,'single_present_class_mean':float(actual[1].mean()),'strict_cpu_backward':True,'zero_identity':True,'real_data_or_gt_used':False,'trainer_sha256':sha(ROOT/'train_fullimage.py')})
print('PASS official hard-IoU including empty classes; strict backward and zero identity',flush=True)
