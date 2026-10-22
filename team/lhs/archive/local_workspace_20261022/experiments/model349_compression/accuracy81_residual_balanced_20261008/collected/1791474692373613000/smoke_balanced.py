"""Synthetic weighted CE equivalence and strict-deterministic CNN backward."""
from pathlib import Path
import json,torch
from train_residual import objective
from residual_model import Residual
from fields_io import sha,write
ROOT=Path(__file__).resolve().parent
results=[]
for device in ['cpu','cuda']:
 torch.manual_seed(81349)
 x=torch.randn(2,4,32,32,device=device,requires_grad=True)
 t=torch.randint(4,(2,32,32),device=device)
 weights=torch.tensor([1.5,.6,.4,1.5],device=device)
 torch.use_deterministic_algorithms(False)
 ref=torch.nn.functional.cross_entropy(x,t,weight=weights)
 reference_gradient=torch.autograd.grad(ref,x)[0]
 torch.use_deterministic_algorithms(True)
 nll=-torch.nn.functional.log_softmax(x,dim=1).gather(1,t[:,None]).squeeze(1)
 pixel_weights=weights[t];value=(nll*pixel_weights).sum()/pixel_weights.sum()
 gradient=torch.autograd.grad(value,x)[0]
 assert torch.allclose(ref,value,rtol=1e-6,atol=1e-7)
 assert torch.allclose(reference_gradient,gradient,rtol=1e-5,atol=1e-8)
 model=Residual().to(device);pred,delta=model(x.detach(),torch.rand(2,1,32,32,device=device))
 assert torch.equal(pred,x.detach()) and torch.count_nonzero(delta)==0
 loss=objective(pred,t,delta,weights);loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 results.append({'device':device,'weighted_ce_value_diff':abs(ref.item()-value.item()),'gradient_max_diff':float((gradient-reference_gradient).abs().max()),'strict_backward_passed':True})
write(ROOT/'smoke_balanced.json',{'results':results,'trainer_sha256':sha(ROOT/'train_residual.py'),'real_data_or_gt_used':False})
print(json.dumps(results),flush=True)
