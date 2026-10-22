"""Read-only eval-BN input moments and explicit running-buffer interpolation."""
import torch
class Moments:
 def __init__(self,model):
  self.model=model;self.layers={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
  assert self.layers and all(not m.training and m.track_running_stats for m in self.layers.values())
  self.original={n:(m.running_mean.detach().clone(),m.running_var.detach().clone()) for n,m in self.layers.items()}
  self.parameter_versions={n:p._version for n,p in model.named_parameters()}
  allowed={n+'.'+s for n in self.layers for s in ['running_mean','running_var']}
  self.other_buffer_versions={n:p._version for n,p in model.named_buffers() if n not in allowed}
  self.stats={};self.handles=[]
 def unchanged_non_bn(self):
  assert self.parameter_versions=={n:p._version for n,p in self.model.named_parameters()}
  assert all(dict(self.model.named_buffers())[n]._version==v for n,v in self.other_buffer_versions.items())
 def restore(self):
  assert not self.handles
  with torch.no_grad():
   for n,m in self.layers.items():m.running_mean.copy_(self.original[n][0]);m.running_var.copy_(self.original[n][1])
  self.assert_original();self.unchanged_non_bn()
 def assert_original(self):
  assert all(torch.equal(m.running_mean,self.original[n][0]) and torch.equal(m.running_var,self.original[n][1]) for n,m in self.layers.items())
 def begin(self):
  self.assert_original();assert not self.handles;self.stats={}
  for n,m in self.layers.items():
   def hook(module,inputs,name=n):
    assert not module.training
    x=inputs[0].detach().double();assert x.ndim>=2 and x.shape[1]==module.num_features
    axes=(0,*range(2,x.ndim));count=x.numel()//x.shape[1];s=x.sum(axes);q=x.square().sum(axes)
    if name not in self.stats:self.stats[name]=[s,q,count]
    else:self.stats[name][0].add_(s);self.stats[name][1].add_(q);self.stats[name][2]+=count
   self.handles.append(m.register_forward_pre_hook(hook))
 def end(self):
  for h in self.handles:h.remove()
  self.handles=[];self.assert_original();self.unchanged_non_bn();assert self.stats
  for s,q,count in self.stats.values():assert count>0 and torch.isfinite(s).all() and torch.isfinite(q).all()
 def apply(self,alpha):
  assert not self.handles and 0<=alpha<=1
  self.restore()
  if alpha==0:return
  with torch.no_grad():
   for n,(s,q,count) in self.stats.items():
    mean=s/count;variance=(q/count-mean.square()).clamp_min(0);m=self.layers[n];orig_mean,orig_var=self.original[n]
    m.running_mean.copy_((1-alpha)*orig_mean.double()+alpha*mean)
    m.running_var.copy_((1-alpha)*orig_var.double()+alpha*variance)
  self.unchanged_non_bn()
