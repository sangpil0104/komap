"""Load cached ImageNet starts, metric and BN freeze checks; no optimizer or GPU."""
import gc,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,write_json
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
sys.path[:0]=[str(ROOT/'source/experiments'),str(ROOT/'source/experiments/historical_scripts')]
import torch,numpy as np
import segmentation_models_pytorch as smp
from train_models_2_to_6 import UNetPlusPlusResNet34
from local_image_metrics import summary
from tqdm import tqdm
torch.set_num_threads(2)
for name in tqdm(['resnet34','efficientnet-b4','tu-hrnet_w48'],desc='CPU ImageNet initialization',unit='model'):
    if name=='resnet34':model=UNetPlusPlusResNet34(pretrained=True)
    else:
        cls=smp.Unet if name=='efficientnet-b4' else smp.UnetPlusPlus
        model=cls(encoder_name=name,encoder_weights='imagenet',in_channels=3,classes=4)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    del model;gc.collect()
pred=np.tile(np.array([0,2,3]),(4,1))
result=summary([pred,pred[:1]],[pred,pred[:1]],['Al3Ni','Primary Si','Al','Eutectic Si'])
assert result['image_mean_miou']==.75 and result['image_class_iou']['Primary Si']==0
# Different image sizes retain equal image weights.
result=summary([pred,np.ones((100,100),int)],[pred,np.zeros((100,100),int)],list('abcd'))
assert result['image_mean_miou']==.375
write_json(ROOT/'checks/models_cpu.json',dict(ok=True,training_started=False,gpu_used=False,
    forward_calls=0,cached_imagenet_encoders=['resnet34','efficientnet-b4','tu-hrnet_w48'],
    empty_union_zero=True,equal_image_weighting=True))
print('PASS: three ImageNet starts load offline on CPU; image mIoU policies verified.')
