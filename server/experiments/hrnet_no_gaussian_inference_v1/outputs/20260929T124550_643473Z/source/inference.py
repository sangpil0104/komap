"""One forward pass can evaluate multiple independent stitching conditions."""
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import DEADLINE

DEFAULT=dict(patch=512,scale=2.,overlap=170,input_mode='bicubic',input_antialias=False,
             restore='bilinear',restore_antialias=False,restore_logits=False,
             window='gaussian',sigma=.125,tta='flip4',tile_merge='prob',tta_merge='prob',
             halo=0,precision='fp32',gamma=1.,contrast=1.,sharpen=0.)
VIEWS={'none':[0],'hflip':[0,1],'vflip':[0,2],'flip4':[0,1,2,3],
       'transpose4':[4,5,6,7],'d4':list(range(8))}


def transform(x, code, inverse=False):
    if inverse and code>=4:x=x.transpose(-2,-1)
    if code%4 & 1:x=x.flip(-1)
    if code%4 & 2:x=x.flip(-2)
    if not inverse and code>=4:x=x.transpose(-2,-1)
    return x


def starts(n,p,stride):
    if stride<=0:raise ValueError('Overlap must be smaller than native tile')
    if n<=p:return [0]
    out=list(range(0,n-p+1,stride))
    if out[-1]!=n-p:out.append(n-p)
    return out


def resize(x,size,mode,antialias=False):
    kw=dict(size=(size,size),mode=mode)
    if mode in ('bilinear','bicubic'):kw.update(align_corners=False,antialias=antialias)
    return F.interpolate(x,**kw)


def window(size,cfg,device):
    if cfg['window']=='uniform':return torch.ones(size,size,device=device)
    if cfg['window']=='hann':
        g=torch.hann_window(size,periodic=False,device=device);w=g[:,None]*g[None,:]
    else:
        c=torch.arange(size,dtype=torch.float32,device=device)-(size-1)/2
        g=torch.exp(-c.square()/(2*(size*cfg['sigma'])**2));w=g[:,None]*g[None,:]
    return (w/w.max()).clamp_min(1e-3)


def forward_key(cfg):
    return tuple(cfg[k] for k in ('patch','scale','overlap','input_mode','input_antialias','halo','precision','gamma','contrast','sharpen'))


@torch.inference_mode()
def predict_group(model,image,arms,device,batch_size,description='inference'):
    cfg=arms[0]['config']
    assert len({forward_key(a['config']) for a in arms})==1
    native=round(cfg['patch']/cfg['scale']);stride=native-cfg['overlap']
    views=sorted({v for a in arms for v in VIEWS[a['config']['tta']]})
    image=image.to(device)
    h0,w0=image.shape[-2:]
    if cfg['halo']:image=F.pad(image[None],(cfg['halo'],)*4,mode='reflect')[0]
    accum={a['id']:None for a in arms}
    work=sum(len(starts(max(transform(image,v).shape[-2],native),native,stride))*len(starts(max(transform(image,v).shape[-1],native),native,stride)) for v in views)
    progress=tqdm(total=work,desc=description,unit='tile',leave=False,mininterval=2)
    for v in views:
        active=[a for a in arms if v in VIEWS[a['config']['tta']]]
        x=transform(image,v);h,w=x.shape[-2:]
        x=F.pad(x[None],(0,max(0,native-w),0,max(0,native-h)),mode='reflect')[0]
        H,W=x.shape[-2:];tiles=[(y,z) for y in starts(H,native,stride) for z in starts(W,native,stride)]
        # Identical merge settings can share tile accumulation across TTA choices.
        unique={}
        arm_keys={}
        for a in active:
            c=a['config'];key=tuple(c[k] for k in ('restore','restore_antialias','restore_logits','window','sigma','tile_merge'))
            arm_keys[a['id']]=key
            if key not in unique:unique[key]=(c,torch.zeros(4,H,W,device=device),torch.zeros(H,W,device=device),window(native,c,device))
        for i in range(0,len(tiles),batch_size):
            if time.time()>=DEADLINE:raise TimeoutError('Requested KST 01:30 deadline reached')
            loc=tiles[i:i+batch_size];batch=torch.stack([x[:,y:y+native,z:z+native] for y,z in loc])
            if native!=cfg['patch']:batch=resize(batch,cfg['patch'],cfg['input_mode'],cfg['input_antialias'])
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if cfg['precision']=='bf16' else nullcontext()
            with ctx:logits=model(batch)
            logits=logits.float();p=logits.softmax(1);restored={}
            for key,(c,total,weights,win) in unique.items():
                rk=(c['restore'],c['restore_antialias'],c['restore_logits'],c['tile_merge'])
                if rk not in restored:
                    q=logits if (c['restore_logits'] or c['tile_merge']=='logit') else p
                    if native!=cfg['patch']:q=resize(q,native,c['restore'],c['restore_antialias'])
                    if c['tile_merge']=='prob':
                        if c['restore_logits']:q=q.softmax(1)
                        elif c['restore']=='bicubic':q=q.clamp_min(0);q=q/q.sum(1,keepdim=True).clamp_min(1e-8)
                    restored[rk]=q
                for (y,z),q in zip(loc,restored[rk]):
                    total[:,y:y+native,z:z+native]+=q*win
                    weights[y:y+native,z:z+native]+=win
            progress.update(len(loc))
        merged={}
        for key,(c,total,weights,win) in unique.items():
            q=(total/weights.clamp_min(1e-8))[:,:h,:w]
            if c['tile_merge']=='logit':q=q.softmax(0)
            q=transform(q,v,inverse=True)
            if cfg['halo']:q=q[:,cfg['halo']:cfg['halo']+h0,cfg['halo']:cfg['halo']+w0]
            merged[key]=q.cpu()
        for a in active:
            q=merged[arm_keys[a['id']]]
            if a['config']['tta_merge']=='log':q=q.clamp_min(1e-8).log()
            accum[a['id']]=q.clone() if accum[a['id']] is None else accum[a['id']]+q
        del unique,merged,batch,logits,p,restored,x
    progress.close()
    for a in arms:
        q=accum[a['id']]/len(VIEWS[a['config']['tta']])
        if a['config']['tta_merge']=='log':q=q.softmax(0)
        accum[a['id']]=q.numpy()
    return accum
