import torch,json
from pathlib import Path
from moments import Moments
torch.manual_seed(81349);model=torch.nn.Sequential(torch.nn.BatchNorm2d(4),torch.nn.ReLU(),torch.nn.BatchNorm2d(4)).eval()
before={n:v.clone() for n,v in model.state_dict().items()};x=torch.randn(2,4,8,12);y=torch.randn(1,4,8,12);capture={}
handles=[]
for n,m in model.named_modules():
 if isinstance(m,torch.nn.BatchNorm2d):handles.append(m.register_forward_pre_hook(lambda m,inp,name=n:capture.setdefault(name,[]).append(inp[0].detach().clone())))
collector=Moments(model);collector.begin()
with torch.no_grad():reference=model(x);model(y)
collector.end()
for h in handles:h.remove()
for n,(s,q,count) in collector.stats.items():
 joined=torch.cat(capture[n]).double();mean=joined.mean((0,2,3));var=joined.var((0,2,3),unbiased=False)
 assert torch.allclose(s/count,mean,rtol=0,atol=1e-12) and torch.allclose(q/count-(s/count).square(),var,rtol=0,atol=1e-12)
assert all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())
collector.apply(0)
with torch.no_grad():assert torch.equal(model(x),reference)
collector.apply(.25)
for n,p in model.named_parameters():assert torch.equal(p,before[n])
collector.restore();assert all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())
Path(__file__).with_name('smoke_moments.json').write_text(json.dumps({'synthetic_moments_match_direct':True,'collector_keeps_original_state':True,'alpha0_output_exact':True,'affine_parameters_unchanged':True,'restore_full_state_exact':True,'real_images_or_gt_used':False},indent=2)+'\n')
print('PASS read-only moments, alpha0 identity, affine unchanged, full state restore',flush=True)
