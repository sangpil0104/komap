"""Original full-image-statistics B4 augmentation, deterministic train70 stream."""
from pathlib import Path
import sys, hashlib, json
import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'Model349_exh14'
sys.path.insert(0, str(SOURCE / 'src'))
import b4_zcdf as b4
PALETTE = np.asarray([[153,127,76],[25,76,153],[76,178,76],[204,204,204]], np.uint8)
SEED = 20261009
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
def objective(logits, y):
    import torch.nn.functional as F
    p = logits.float().softmax(1)
    gt = F.one_hot(y, 4).permute(0,3,1,2).float()
    inter = (p*gt).sum((0,2,3)); den = p.sum((0,2,3))+gt.sum((0,2,3))
    dice = 1-((2*inter+1)/(den+1)).mean()
    return F.cross_entropy(logits.float(), y)+dice
class Dataset:
    def __init__(self):
        d=Path('/workspace/komap_model1/data/train')
        files=sorted((d/'images').glob('*_image.png')); assert len(files)==70
        self.rows=[]; self.gray=[]; self.labels=[]; self.scales=[]; self.stats=[]; self.cdfs=[]
        self.transform=A.Compose([
            A.Affine(scale=(.7,1.4),rotate=(-180,180),border_mode=cv2.BORDER_REFLECT_101,p=.7),
            A.PadIfNeeded(1536,1536,border_mode=cv2.BORDER_REFLECT_101),A.RandomCrop(1536,1536),A.D4(),
            A.RandomBrightnessContrast(brightness_limit=.3,contrast_limit=.4,p=.8),
            A.RandomGamma(gamma_limit=(60,160),p=.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3,5)),A.GaussNoise(std_range=(.01,.05))],p=.3)])
        for p in files:
            m=d/'masks'/p.name.replace('_image.png','_mask.png'); assert m.is_file()
            g=b4.load_gray(p); rgb=np.asarray(Image.open(m).convert('RGB'))
            label=np.full(rgb.shape[:2],255,np.uint8)
            for c,color in enumerate(PALETTE): label[np.all(rgb==color,axis=-1)]=c
            assert np.all(label<4) and label.shape==g.shape
            scale=4000/g.shape[1]; full=b4.resize_gray(g,scale); values=full.astype(np.float32)/255
            stats=(float(values.mean()),float(values.std()))
            hist=np.bincount(full.reshape(-1),minlength=256).astype(np.float64)
            cdf=(np.cumsum(hist)/hist.sum()).astype(np.float32)
            self.gray.append(g); self.labels.append(label); self.scales.append(scale); self.stats.append(stats); self.cdfs.append(cdf)
            self.rows.append({'name':p.name.removesuffix('_image.png'),'image':str(p),'mask':str(m),'image_sha256':sha(p),'mask_sha256':sha(m),'width4000_mean':stats[0],'width4000_std':stats[1],'cdf_sha256':hashlib.sha256(cdf.tobytes()).hexdigest()})
    def sample(self, index):
        seed=SEED+index; rng=np.random.default_rng(seed); self.transform.set_random_seed(seed)
        i=int(rng.integers(70)); g=self.gray[i]; m=self.labels[i]; scale=self.scales[i]
        size=int(1536*2.1/scale); h,w=g.shape; top=left=0
        if h>size and w>size:
            top=int(rng.integers(h-size+1)); left=int(rng.integers(w-size+1))
            g=g[top:top+size,left:left+size]; m=m[top:top+size,left:left+size]
        if scale!=1:
            g=b4.resize_gray(g,scale); m=cv2.resize(m,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_NEAREST)
        out=self.transform(image=g,mask=m)
        x=b4.make_input(out['image'],'image_global','zcdf3',None,self.stats[i],self.cdfs[i])
        assert x.shape==(3,1536,1536) and out['mask'].shape==(1536,1536) and np.isfinite(x).all()
        return torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(np.ascontiguousarray(out['mask'])).long(),{'index':index,'seed':seed,'image_index':i,'image':self.rows[i]['name'],'top':top,'left':left}
