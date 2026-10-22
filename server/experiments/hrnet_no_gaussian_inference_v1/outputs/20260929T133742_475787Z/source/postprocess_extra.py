"""Label-independent, weak edge-aware probability message passing."""
import cv2
import numpy as np

import postprocess as base

ORIGINAL_APPLY=base.apply


def bilateral_messages(p,image,cfg):
    radius=cfg.get('radius',3);space=cfg.get('space_sigma',1.5);color=cfg['color_sigma']
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    h,w=gray.shape;pad=np.pad(gray,radius,mode='reflect');neighbors=[];denominator=np.zeros((h,w),np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dy==dx==0:continue
            shifted=pad[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
            weight=np.exp(-((gray-shifted)**2)/(2*color*color)-(dy*dy+dx*dx)/(2*space*space)).astype(np.float32)
            denominator+=weight;neighbors.append((dy,dx,weight))
    denominator=np.maximum(denominator,1e-10)
    for dy,dx,weight in neighbors:weight/=denominator
    unary=np.log(np.maximum(p,1e-8));q=p.copy()
    for _ in range(cfg.get('iterations',3)):
        padded=np.pad(q,((0,0),(radius,radius),(radius,radius)),mode='reflect');message=np.zeros_like(q)
        for dy,dx,weight in neighbors:
            message+=padded[:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]*weight[None]
        logits=unary+cfg['strength']*message;logits-=logits.max(0,keepdims=True)
        q=np.exp(logits);q/=q.sum(0,keepdims=True)
    return q


def apply(p,image,change):
    if change['kind']=='bilateral_messages':return bilateral_messages(p,image,change)
    return ORIGINAL_APPLY(p,image,change)


if __name__=='__main__':
    base.apply=apply
    base.main()
