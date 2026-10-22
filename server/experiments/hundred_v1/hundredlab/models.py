"""Small, explicit research-inspired variants; not official paper replicas."""
import torch
from torch import nn
from torch.nn import functional as F
from alphase.models.cnn import PhaseCNN
from alphase.models.hybrid import ResidualBlock
from alphase.data.palette import PALETTE
from alphase.data.splits import select_rows
from alphase.utils import sha256


def zero(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None: nn.init.zeros_(module.bias)
    return module


class Attention(nn.Module):
    def __init__(self,c,kind,variant=1):
        super().__init__(); self.kind=kind
        if kind=='eca':
            self.channel=zero(nn.Conv1d(1,1,3 if variant==1 else 5,padding=1 if variant==1 else 2,bias=False))
        else:
            hidden=max(8,c//(16 if variant==1 else 8))
            self.channel=nn.Sequential(nn.Conv2d(c,hidden,1),nn.GELU(),zero(nn.Conv2d(hidden,c,1)))
        if kind=='cbam': self.spatial=zero(nn.Conv2d(2,1,7 if variant==1 else 3,padding=3 if variant==1 else 1))
    def forward(self,x):
        avg=x.mean((2,3),keepdim=True)
        if self.kind=='eca': gate=self.channel(avg.flatten(2).transpose(1,2)).transpose(1,2).unsqueeze(-1)
        else:
            gate=self.channel(avg)
            if self.kind=='cbam': gate=gate+self.channel(x.amax((2,3),keepdim=True))
        x=x*(2*gate.sigmoid())
        if self.kind=='cbam': x=x*(2*self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)).sigmoid())
        return x


class Context(nn.Module):
    def __init__(self,c,kind,variant=1,width=None,rates=None):
        super().__init__(); self.kind=kind
        width=width or (32 if variant==1 else 64)
        self.reduce=nn.Sequential(nn.Conv2d(c,width,1),nn.GroupNorm(8,width),nn.GELU())
        if kind=='aspp':
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=d,dilation=d,groups=width),nn.GELU())
                for d in (rates or ([1,2,4,8] if variant==1 else [1,3,6,9]))])
            out=4*width
        elif kind=='ppm':
            self.bins=[1,2,4] if variant==1 else [1,3,6]
            self.branches=nn.ModuleList([nn.Conv2d(width,width,1) for _ in self.bins]);out=4*width
        else:
            k=7 if variant==1 else 11
            if kind=='large_kernel': body=nn.Sequential(nn.Conv2d(width,width,k,padding=k//2,groups=width),nn.GELU(),nn.Conv2d(width,width,1))
            elif kind=='axial': body=nn.Sequential(nn.Conv2d(width,width,(1,k),padding=(0,k//2),groups=width),nn.GELU(),nn.Conv2d(width,width,(k,1),padding=(k//2,0),groups=width))
            elif kind=='channel_adapter': body=nn.Sequential(nn.Conv2d(width,width,1),nn.GELU())
            else: body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1))
            self.body=body;out=width
        self.project=zero(nn.Conv2d(out,c,1))
    def forward(self,x):
        z=self.reduce(x)
        if self.kind=='aspp': z=torch.cat([b(z) for b in self.branches],1)
        elif self.kind=='ppm': z=torch.cat([z,*[F.interpolate(b(F.adaptive_avg_pool2d(z,n)),size=z.shape[-2:],mode='bilinear',align_corners=False) for n,b in zip(self.bins,self.branches)]],1)
        else:z=self.body(z)
        return x+self.project(z)


class SkipGate(nn.Module):
    def __init__(self,c,g,width):
        super().__init__()
        self.x=nn.Conv2d(c,width,1);self.g=nn.Conv2d(g,width,1)
        self.out=zero(nn.Conv2d(width,1,1))
    def forward(self,x,g):
        z=self.x(x)+F.interpolate(self.g(g),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return x*(2*self.out(F.gelu(z)).sigmoid())


class WarmCNN(PhaseCNN):
    def __init__(self,options):
        super().__init__(3)
        self.kind=options.get('adapter','none');v=options.get('variant',1)
        self.additions=nn.ModuleDict()
        if self.kind in ('se','eca','cbam'):
            self.additions['context']=Attention(384,self.kind,v)
        elif self.kind in ('aspp','ppm','large_kernel','axial','bottleneck','channel_adapter'):
            self.additions['context']=Context(384,self.kind,v,options.get('adapter_width'),options.get('rates'))
        elif self.kind=='skip_gate':
            self.additions['gates']=nn.ModuleList([SkipGate(c,g,8*v) for c,g in [(256,384),(128,256),(64,128),(32,64)]])
        elif self.kind=='decoder_refine':
            self.additions['refine']=Context(32,'bottleneck',v)
        elif self.kind=='highres':
            self.additions['side']=nn.Sequential(nn.Conv2d(3,8*v,3,padding=1),nn.GELU(),zero(nn.Conv2d(8*v,4,1)))
        elif self.kind!='none':raise ValueError(self.kind)
    def forward(self,detail,dino=None,box=None):
        c1,c2,c4,c8,x=self.detail(detail)
        if 'context' in self.additions:x=self.additions['context'](x)
        d4=None
        for i,(skip,block) in enumerate(zip((c8,c4,c2,c1),(self.up8,self.up4,self.up2,self.up1))):
            if 'gates' in self.additions:skip=self.additions['gates'][i](skip,x)
            x=block(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
            if i==1:d4=x
        if 'refine' in self.additions:x=self.additions['refine'](x)
        logits=self.classifier(x)
        if 'side' in self.additions:logits=logits+self.additions['side'](detail)
        return dict(logits=logits,aux_logits=self.aux_head(d4),boundary_logits=self.boundary_head(x))


class EfficientBlock(nn.Module):
    def __init__(self,cin,cout,kind):
        super().__init__();self.kind=kind
        self.project=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        if kind=='convnext':
            self.body=nn.Sequential(nn.Conv2d(cout,cout,7,padding=3,groups=cout),nn.GroupNorm(1,cout),
                nn.Conv2d(cout,4*cout,1),nn.GELU(),nn.Conv2d(4*cout,cout,1))
        else:
            self.body=nn.Sequential(nn.Conv2d(cout,cout,3,padding=1,groups=cout),nn.Conv2d(cout,cout,1),
                nn.GroupNorm(8,cout),nn.GELU(),nn.Conv2d(cout,cout,3,padding=1,groups=cout),nn.Conv2d(cout,cout,1),nn.GroupNorm(8,cout))
    def forward(self,x):
        x=self.project(x)
        return F.gelu(x+self.body(x))


class ScratchUNet(nn.Module):
    def __init__(self,options):
        super().__init__();w=options['width'];kind=options['block'];widths=[w,2*w,4*w,8*w,12*w]
        def block(a,b):
            if kind=='residual':return ResidualBlock(a,b)
            if kind=='attention':return nn.Sequential(ResidualBlock(a,b),Attention(b,'se'))
            return EfficientBlock(a,b,kind)
        self.encoder=nn.ModuleList([block(3,w),*[nn.Sequential(nn.Conv2d(a,b,2,stride=2),block(b,b)) for a,b in zip(widths[:-1],widths[1:])]])
        self.decoder=nn.ModuleList([block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.classifier=nn.Conv2d(w,4,1);self.aux_head=nn.Conv2d(4*w,4,1);self.boundary_head=nn.Conv2d(w,1,1)
    def forward(self,detail,dino=None,box=None):
        features=[];x=detail
        for layer in self.encoder:x=layer(x);features.append(x)
        aux=None
        for i,(skip,block) in enumerate(zip(features[-2::-1],self.decoder)):
            x=block(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
            if i==1:aux=x
        return dict(logits=self.classifier(x),aux_logits=self.aux_head(aux),boundary_logits=self.boundary_head(x))


def build_model(cfg,load_pretrained=True):
    model=ScratchUNet(cfg['model']) if cfg['model']['architecture']=='scratch' else WarmCNN(cfg['model'])
    ref=cfg['hundred']
    if load_pretrained and ref['parent_checkpoint']:
        path=ref['parent_checkpoint']
        if sha256(path)!=ref['parent_sha256']:raise ValueError('Parent checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        train,valid=select_rows(cfg,'official')
        if set(state['train_ids'])!={r['image_id'] for r in train} or set(state['validation_ids'])!={r['image_id'] for r in valid}:
            raise ValueError('Parent split mismatch')
        if state['palette']!=PALETTE.tolist() or state['resolved_config']['data']['classes']!=cfg['data']['classes']:
            raise ValueError('Parent class mapping mismatch')
        selected=state['ema_model'] if state['selected_weights']=='ema' else state['model']
        missing,unexpected=model.load_state_dict(selected,strict=False)
        if set(missing)!={n for n in model.state_dict() if n.startswith('additions.')} or unexpected:
            raise ValueError('Unexpected parent transfer')
    configure_trainable(model,cfg,cfg['train']['stages'][0])
    return model


def configure_trainable(model,cfg,stage):
    policy=stage.get('policy','all')
    for name,p in model.named_parameters():
        enabled=(policy=='all' or policy=='new' and name.startswith('additions.')
                 or policy=='decoder' and not name.startswith(('detail.','encoder.'))
                 or policy=='head' and name.startswith('classifier.'))
        if name.startswith('aux_head.') and not cfg['loss']['auxiliary']:enabled=False
        if name.startswith('boundary_head.') and not cfg['loss']['boundary']:enabled=False
        p.requires_grad_(enabled)
    if not any(p.requires_grad for p in model.parameters()):raise ValueError('No trainable parameters')
