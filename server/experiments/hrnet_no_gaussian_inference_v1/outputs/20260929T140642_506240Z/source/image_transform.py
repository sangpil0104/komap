"""Label-independent whole-image intensity transforms before normalization."""
import cv2
import numpy as np


def apply(image, cfg):
    strength=float(cfg.get('strength',1.))
    if not 0<=strength<=1:raise ValueError('strength must be in [0,1]')
    if strength==0:return image.copy()
    x=image.astype(np.float32)
    if cfg['kind']=='brightness':
        altered=x+cfg['offset']
    elif cfg['kind']=='gamma':
        altered=(((x/255)**cfg['gamma']-.5)+.5)*255
    elif cfg['kind']=='percentile':
        lo,hi=np.percentile(x,[cfg['percentile'],100-cfg['percentile']])
        altered=(x-lo)*255/max(hi-lo,1.)
    elif cfg['kind']=='clahe':
        lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=cfg['clip'],tileGridSize=(cfg.get('grid',8),)*2).apply(lab[:,:,0])
        altered=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB).astype(np.float32)
    elif cfg['kind']=='illumination':
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)
        field=cv2.GaussianBlur(gray,(0,0),cfg['sigma'],borderType=cv2.BORDER_REFLECT_101)
        altered=x*(float(gray.mean())/np.maximum(field,1.))[...,None]
    else:raise ValueError(cfg)
    return np.rint((1-strength)*x+strength*altered).clip(0,255).astype(np.uint8)
