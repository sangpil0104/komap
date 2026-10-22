import math
import torch
from torch.nn import functional as F
from alphase.losses.segmentation import segmentation_loss as original_loss, boundary_target, lovasz_present


def segmentation_loss(outputs,target,weights):
    custom=any(weights.get(k) for k in ('class_weights','focal_gamma','label_smoothing','ohem_fraction','edge_weight','overlap_kind'))
    if not custom:return original_loss(outputs,target,weights)
    logits=outputs['logits'].float()
    base,_=original_loss(outputs,target,{**weights,'ce':0.,'dice':0.,'lovasz':0.})
    boundaries,_=boundary_target(target)
    parts={k:[] for k in ('ce','dice','lovasz')}
    for z,t,edge in zip(logits,target,boundaries):
        valid=t!=255;z=z[:,valid];t=t[valid];edge=edge[valid]
        if not len(t):raise ValueError('No valid target pixels')
        cw=z.new_tensor(weights['class_weights']) if weights.get('class_weights') else None
        ce=F.cross_entropy(z.T,t,weight=cw,label_smoothing=weights.get('label_smoothing',0.),reduction='none')
        if weights.get('focal_gamma'):
            pt=z.softmax(0).gather(0,t[None])[0]
            # Fractional gamma has an infinite derivative at pt=1 without this floor.
            ce=ce*(1-pt).clamp_min(1e-7).pow(weights['focal_gamma'])
        edge_factor=1+weights.get('edge_weight',0.)*edge
        ce=ce*edge_factor
        fraction=weights.get('ohem_fraction',1.)
        if fraction<1:ce=ce.topk(max(1,math.ceil(len(ce)*fraction))).values
        if cw is not None and fraction==1 and not weights.get('focal_gamma'):
            ce=ce.sum()/(cw[t]*edge_factor).sum().clamp_min(1e-8)
        else:ce=ce.mean()
        p=z.softmax(0);present=t.unique();overlap=[]
        if weights.get('overlap_kind')=='generalized_dice':
            truth=F.one_hot(t,4).T.float()[present];prob=p[present]
            w=truth.sum(1).clamp_min(1).pow(-2)
            dice=1-(2*(w*(prob*truth).sum(1)).sum()+1e-6)/((w*(prob.sum(1)+truth.sum(1))).sum()+1e-6)
        else:
            for c in present:
                truth=(t==c).float();tp=(p[c]*truth).sum()
                if weights.get('overlap_kind')=='tversky':
                    alpha=weights.get('tversky_alpha',.5)
                    denom=tp+alpha*(p[c]*(1-truth)).sum()+(1-alpha)*((1-p[c])*truth).sum()
                    overlap.append(1-(tp+1e-6)/(denom+1e-6))
                else:overlap.append(1-(2*tp+1e-6)/(p[c].sum()+truth.sum()+1e-6))
            dice=torch.stack(overlap).mean()
        parts['ce'].append(ce);parts['dice'].append(dice)
        parts['lovasz'].append(lovasz_present(p,t) if weights['lovasz'] else z.sum()*0)
    parts={k:torch.stack(v).mean() for k,v in parts.items()}
    return base+sum(weights[k]*v for k,v in parts.items()),{k:v.detach() for k,v in parts.items()}
