"""CPU equivalence check against the frozen original predictor, asymmetric tiled input."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/workspace/model349_compress/audit_20261008')
sys.path.insert(0,str(root/'bundles/lite15/src')); sys.path.insert(0,str(root))
from style_tta_search import predict_array as original
from style_tta_batched import predict_array as batched
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer=torch.nn.Conv2d(3,4,1); self.calls=[]
    def forward(self,x):
        self.calls.append(int(x.shape[0])); return self.layer(x)
torch.set_num_threads(2); torch.manual_seed(32)
model=Model().eval(); image=np.random.default_rng(349).integers(0,256,(23,27,3),dtype=np.uint8)
a=original(model,image,32,16,'cpu',False,1,'d4','gaussian'); old_calls=model.calls.copy(); model.calls=[]
b=batched(model,image,32,16,'cpu',False,1,'d4','gaussian')
assert a.shape==b.shape==(4,23,27)
error=float(np.max(np.abs(a-b))); changed=int(np.count_nonzero(a.argmax(0)!=b.argmax(0)))
assert error<1e-6 and changed==0
assert len(model.calls)*4==len(old_calls) and all(x==4 for x in model.calls)
result={'max_probability_abs_error':error,'changed_argmax_pixels':changed,'original_forward_batch_sizes':old_calls,'batched_forward_batch_sizes':model.calls,'input_shape':list(image.shape)}
(root/'batched_views_check.json').write_text(json.dumps(result,indent=2)+'\n'); print(result)
