"""Gray-guidance local linear filtering, ECCV2010 equations 5,6,8.

Independent implementation. Reflected borders differ from clipped windows.
"""
import cv2
import numpy as np
def box(a,r):
 return cv2.boxFilter(a,cv2.CV_32F,(2*r+1,2*r+1),normalize=True,borderType=cv2.BORDER_REFLECT_101)
def guided(guide,field,r,eps):
 guide=np.asarray(guide,dtype=np.float32);field=np.asarray(field,dtype=np.float32)
 assert guide.ndim==2 and field.shape==guide.shape and r>=0 and eps>0
 if not r:return field.copy()
 mi=box(guide,r);mp=box(field,r)
 variance=np.maximum(box(guide*guide,r)-mi*mi,0.)
 a=(box(guide*field,r)-mi*mp)/(variance+eps);b=mp-a*mi
 return box(a,r)*guide+box(b,r)
def refine(logits,guide,r,eps,blend):
 if not blend:return logits.copy()
 value=np.stack([guided(guide,field,r,eps) for field in logits]);return (1-blend)*logits+blend*value
def synthetic_smoke():
 rng=np.random.default_rng(818);i=rng.random((15,17),dtype=np.float32);p=rng.normal(size=(15,17)).astype(np.float32)
 assert np.array_equal(guided(i,p,0,.001),p)
 for r in [1,2,4]:
  for eps in [.0001,.001,.01]:
   q=guided(i,np.full_like(i,2.5),r,eps);assert np.max(np.abs(q-2.5))<2e-5
   q=guided(np.zeros_like(i),p,r,eps);assert np.allclose(q,box(box(p,r),r),atol=1e-6)
   assert np.isfinite(guided(i,p,r,eps)).all()
 assert np.array_equal(refine(p[None].repeat(4,0),i,2,.001,0),p[None].repeat(4,0))
if __name__=='__main__':synthetic_smoke();print('GUIDED_RADIUS_ZERO_CONSTANT_FIELD_CONSTANT_GUIDE_FINITE_SMOKE_PASSED')
