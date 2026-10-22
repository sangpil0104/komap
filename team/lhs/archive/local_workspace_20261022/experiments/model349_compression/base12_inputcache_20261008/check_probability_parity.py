import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,numpy as np
r=Path('/workspace/model349_compress/base12_inputcache_20261008')
a=Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12');b=r/'runs/cache12_valid20'
old=json.loads((a/'input_manifest.json').read_text());new=json.loads((b/'input_manifest.json').read_text())
assert {x['name']:x['sha256'] for x in old['images']}=={x['name']:x['sha256'] for x in new['images']}
result={}
for c in ['326','329','336','269']:
 rows=[]
 for row in old['images']:
  n=row['name'];x=np.load(a/'probabilities'/c/(n+'.npy'));y=np.load(b/'probabilities'/c/(n+'.npy'));assert x.shape==y.shape
  changed=int(np.count_nonzero(x!=y));rows.append({'image':n,'exact_equal':changed==0,'changed_values':changed,'values':int(x.size),'max_abs_error':float(np.max(np.abs(x.astype(np.float32)-y.astype(np.float32)))) if changed else 0})
 result[c]={'all20_exact_equal':all(x['exact_equal'] for x in rows),'images':rows}
(r/'component_probability_parity.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v['all20_exact_equal'] for k,v in result.items()})
