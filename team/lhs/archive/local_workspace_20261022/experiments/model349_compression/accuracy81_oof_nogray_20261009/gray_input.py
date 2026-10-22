import numpy as np
MODE='zero'
def prepare_gray(gray):
 a=np.asarray(gray,dtype=np.float32)
 if MODE=='zero':return np.full_like(a,.5)
 mean=float(np.mean(a,dtype=np.float64));std=float(np.std(a,dtype=np.float64))
 if std<1e-6:return np.full_like(a,.5)
 return (np.clip((a-mean)/std,-3,3)/6+.5).astype(np.float32)
