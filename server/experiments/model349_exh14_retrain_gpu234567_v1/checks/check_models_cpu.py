"""Load actual ancestors and cached ImageNet models on CPU, without forward/backward."""
import gc
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, write_json

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
sys.path[:0]=[str(ROOT/'source/experiments'),str(ROOT/'source/experiments/historical_scripts')]
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
torch.set_num_threads(2)
from encoder_structure_search import build_model
from model253_highres_scale_finetune import Stride1NativeStemASPP

parent98=ROOT/'initial_weights/model98_hard_negative/best.pt'
model=build_model('aspp',False,str(parent98))
payload=torch.load(parent98,map_location='cpu',weights_only=False)
missing,unexpected=model.load_state_dict(payload['model'],strict=False)
assert not unexpected
assert all(name.startswith('context.') for name in missing),missing
del payload,model;gc.collect()
model=Stride1NativeStemASPP()
model.load_state_dict(torch.load(ROOT/'initial_weights/model257_full_image_stats_1x/best.pt',map_location='cpu',weights_only=False)['model'])
del model;gc.collect()
for encoder in tqdm(['efficientnet-b4','tu-hrnet_w48'],desc='CPU ImageNet cache load',unit='encoder'):
    cls=smp.Unet if encoder=='efficientnet-b4' else smp.UnetPlusPlus
    model=cls(encoder_name=encoder,encoder_weights='imagenet',in_channels=3,classes=4)
    del model;gc.collect()
write_json(ROOT/'checks/models_cpu.json',dict(ok=True,training_started=False,forward_calls=0,
    gpu_used=False,ancestors=['Model98','Model257'],model109_new_context_parameters=missing,
    cached_imagenet_encoders=['efficientnet-b4','tu-hrnet_w48']))
print('PASS: Model98/257 parents and offline ImageNet encoders load on CPU; no forward or training.')
