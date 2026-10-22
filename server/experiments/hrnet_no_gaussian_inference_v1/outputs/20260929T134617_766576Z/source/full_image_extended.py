"""Full-image interpolation and phase ablations; fixed learned parameters."""
from contextlib import nullcontext
import time

import torch
import torch.nn.functional as F

from common import DEADLINE
from inference import VIEWS, transform


@torch.inference_mode()
def predict(model,image,device,scale=2.,tta='d4',precision='fp32',input_mode='bicubic',
            restore_mode='bilinear',input_align_corners=False,restore_align_corners=False,
            padding_mode='reflect',padding_position='bottom_right',halo=0,restore_logits=False,
            temperature=1.,tta_merge='prob',model_halo=0):
    h0,w0=image.shape[-2:]
    if halo:image=F.pad(image[None],(halo,)*4,mode='reflect')[0]
    total=None
    for view in VIEWS[tta]:
        if time.time()>=DEADLINE:raise TimeoutError('Requested deadline reached')
        x=transform(image,view).unsqueeze(0).to(device);h,w=x.shape[-2:]
        size=(round(h*scale),round(w*scale))
        if scale!=1:x=F.interpolate(x,size=size,mode=input_mode,align_corners=input_align_corners)
        H,W=x.shape[-2:]
        if model_halo:x=F.pad(x,(model_halo,)*4,mode='reflect')
        ph,pw=x.shape[-2:];dh,dw=(-ph)%32,(-pw)%32
        if padding_position=='bottom_right':top,left=0,0
        elif padding_position=='center':top,left=dh//2,dw//2
        else:raise ValueError(padding_position)
        x=F.pad(x,(left,dw-left,top,dh-top),mode=padding_mode)
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
        with ctx:logits=model(x)
        logits=logits.float()[...,top+model_halo:top+model_halo+H,left+model_halo:left+model_halo+W]
        if temperature!=1:logits=logits/temperature
        p=logits if restore_logits else logits.softmax(1)
        if (H,W)!=(h,w):
            kwargs=dict(size=(h,w),mode=restore_mode)
            if restore_mode in ('bilinear','bicubic'):kwargs['align_corners']=restore_align_corners
            p=F.interpolate(p,**kwargs)
        if restore_logits:p=p.softmax(1)
        elif restore_mode=='bicubic':p=p.clamp_min(0);p=p/p.sum(1,keepdim=True).clamp_min(1e-8)
        p=transform(p[0],view,inverse=True)
        if halo:p=p[:,halo:halo+h0,halo:halo+w0]
        p=p.cpu()
        if tta_merge=='log':p=p.clamp_min(1e-8).log()
        total=p if total is None else total+p
    total=total/len(VIEWS[tta])
    if tta_merge=='log':total=total.softmax(0)
    return total.numpy()
