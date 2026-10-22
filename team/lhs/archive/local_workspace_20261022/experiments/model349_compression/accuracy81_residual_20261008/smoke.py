import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,pickle
import numpy as np
from PIL import Image
from residual_model import smoke
from fields_io import api,finish,SOURCE
smoke()
run=SOURCE.parent/'exh14_base_search_20261008/runs/control14_valid20';m=json.loads((run/'input_manifest.json').read_text());row=m['images'][0];name=row['name'];entry,fusion,precleanup,_=api()
with (run/'probabilities/base'/f'{name}.pkl').open('rb') as f:item=pickle.load(f)
sources={c:np.load(run/'probabilities'/c/f'{name}.npy').astype(np.float32) for c in entry.active_components(m['config']) if c!='base'}
item,mixed,s,weights=fusion(item,sources,m['config']);lp=precleanup(item,mixed,s,weights,True)
assert lp.shape==(4,row['size'][1],row['size'][0]) and np.isfinite(lp).all()
assert np.array_equal(finish(lp,m['config']),np.array(Image.open(run/'predicted_labels'/f'{name}.png')))
print('RESIDUAL_SYNTHETIC_ZERO_IDENTITY_AND_ACTUAL_PARENT_PRE_CLEANUP_MASK_EXACT_PASSED')
