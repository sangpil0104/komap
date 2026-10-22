"""Image-only full-image raw/z-score/CDF RGB8 preprocessing for N3."""
from pathlib import Path
import numpy as np
from PIL import Image

def encode(path):
 gray=np.asarray(Image.open(path).convert('L'))
 x=gray.astype(np.float32)/255.0
 z=(x-x.mean())/(x.std()+1e-6)
 channel=np.rint(np.clip((z/4.0)*.5+.5,0.,1.)*255.).astype(np.uint8)
 hist=np.bincount(gray.reshape(-1),minlength=256)
 cdf=np.cumsum(hist)/max(float(hist.sum()),1.)
 equalized=np.rint(cdf[gray]*255.).astype(np.uint8)
 return np.stack([gray,channel,equalized],axis=-1)

def prepare(input_dir,output_dir):
 output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
 for path in sorted(Path(input_dir).iterdir()):
  if path.suffix.lower() not in ['.png','.jpg','.jpeg','.tif','.tiff','.bmp']:continue
  Image.fromarray(encode(path)).save(output_dir/(path.stem+'.png'))
 return output_dir
