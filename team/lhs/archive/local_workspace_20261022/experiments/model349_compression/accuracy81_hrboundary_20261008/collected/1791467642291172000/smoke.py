"""CPU checks for analytic loss, finite gradients, paired crops and labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,random
import numpy as np
import torch
from loss import lovasz_softmax,objective
# Four pixels, GT all class0. Present class Lovasz=.55, three absent maxima=.3 each.
p=torch.tensor([[.8,.5,.1,.4],[.1,.2,.3,.2],[.05,.1,.3,.2],[.05,.2,.3,.2]])
x=p.log().reshape(1,4,2,2).requires_grad_();y=torch.zeros(1,2,2,dtype=torch.long)
v=lovasz_softmax(x,y);assert abs(v.item()-.3625)<1e-6,v
v.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
z=torch.full((1,4,2,2),-20.);target=torch.tensor([[[0,1],[2,3]]]);z.scatter_(1,target[:,None],20.)
assert lovasz_softmax(z,target).item()<1e-6
for arm in ['control','lovasz']:
 x=torch.randn(2,4,7,9,requires_grad=True);y=torch.randint(0,4,(2,7,9));loss=objective(x,y,arm);loss.backward();assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
# No data split access: synthetic dataset exercises actual crop/augmentation stream.
from train import PairedDataset
from training_dataset import build_train_transform
h=w=320;rng=np.random.default_rng(19)
image=rng.integers(0,256,(h,w,3),dtype=np.uint8);mask=rng.integers(0,4,(h,w),dtype=np.uint8)
def ds():
 d=object.__new__(PairedDataset);d.images=[image];d.class_masks=[mask];d.minority_coords=[{1:np.array([[30,30],[100,100]]),3:np.array([[200,200]])}];d.crop_size=256;d.patch_size=512;d.input_scale=2.;d.positive_sample_prob=.5;d.transform=build_train_transform();d.rng=random.Random(81336);d.epoch=1;return d
one,two=ds(),ds()
for i in [0,5,11]:
 a,b=one[i],two[i];assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]);assert a[0].shape==(3,512,512) and a[1].shape==(512,512) and set(a[1].unique().tolist())<=set(range(4))
from training_palette import PALETTE,rgb_to_class
assert np.array_equal(rgb_to_class(PALETTE[np.array([[0,1],[2,3]])]),np.array([[0,1],[2,3]]))
# Frozen released checkpoint loads without any new pretrained download.
from train import SOURCE
from komap_seg.model import build_model
saved=torch.load(SOURCE/'weights/model336_hrnet_best/last.pt',map_location='cpu',weights_only=False)
model=build_model(**saved['model_config'],encoder_weights=None);model.load_state_dict(saved['state_dict']);model.eval();torch.set_num_threads(1)
with torch.no_grad():pred=model(torch.zeros(1,3,128,128))
assert pred.shape==(1,4,128,128) and torch.isfinite(pred).all()
print('LOSS_PAIRED_DATASET_PALETTE_FROZEN_MODEL_SMOKE_PASSED')

# Boundary implementation matches the actual original training definitions.
import ast,hashlib,json
from training_boundary import BoundaryHeadLoss,gt_boundary_band
from training_boundary import forward_with_boundary
original=Path('/workspace/komap_model1/model336_hrnet_best/komap_seg/losses.py')
def defs(path):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['gt_boundary_band','_soft_dice','BoundaryHeadLoss']}
assert defs(original)==defs(Path(__file__).parent/'training_boundary.py')
original_model=Path('/workspace/komap_model1/model336_hrnet_best/komap_seg/model.py')
def forward_ast(path):
 return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='forward_with_boundary'),include_attributes=False)
assert forward_ast(original_model)==forward_ast(Path(__file__).parent/'training_boundary.py')
y=torch.zeros(1,128,128,dtype=torch.long);y[:,64:]=1
band=gt_boundary_band(y);assert band.sum().item()==256 and torch.equal(band[:,63:65],torch.ones(1,2,128))
empty=torch.zeros(1,8,8,dtype=torch.long);logit=torch.zeros(1,1,8,8,requires_grad=True);b=BoundaryHeadLoss()(logit,empty);b.backward();assert torch.isfinite(b) and torch.isfinite(logit.grad).all()
seg,bound=forward_with_boundary(model,torch.zeros(1,3,128,128));assert bound.shape==(1,1,128,128)
v=objective(seg,y,'control')+BoundaryHeadLoss()(bound,y);v.backward()
assert torch.isfinite(v) and model.boundary_head.weight.grad is not None and torch.isfinite(model.boundary_head.weight.grad).all()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('BOUNDARY_ORIGINAL_AST_STEP_BAND_EMPTY_FINITE_FROZEN_PARENT_BACKWARD_PASSED')
