"""Read-only canonical audit after interrupted packing; never writes weights."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib,json,zipfile
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT=Path('/workspace/model349_compress/base12_resident_packed_20261008')
BASE=Path('/workspace/model349_compress/audit_20261008/bundles/cpu2_fp16_12')
PACKED=ROOT/'bundles/packed_fast12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def leaves(o,key='root'):
    if isinstance(o,torch.Tensor):return {key:o}
    if isinstance(o,dict):return {p:t for k,v in o.items() for p,t in leaves(v,key+'/'+str(k)).items()}
    if isinstance(o,(list,tuple)):return {p:t for k,v in enumerate(o) for p,t in leaves(v,key+'/'+str(k)).items()}
    return {}
rows=[]
before=json.loads((ROOT/'frozen_before.json').read_text())
for row in json.loads((BASE/'checkpoint_manifest.json').read_text())['checkpoints']:
    a,b=BASE/row['path'],PACKED/row['path']
    assert sha(a)==before[row['path']] and a.stat().st_size==row['bytes']
    with zipfile.ZipFile(a) as z:ah={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    with zipfile.ZipFile(b) as z:bh={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    assert ah==bh
    x,y=leaves(torch.load(a,map_location='cpu',weights_only=False)),leaves(torch.load(b,map_location='cpu',weights_only=False))
    assert x and x.keys()==y.keys()
    assert all(x[k].shape==y[k].shape and x[k].dtype==y[k].dtype and torch.equal(x[k],y[k]) for k in x)
    rows.append({'path':row['path'],'original_base_bytes':a.stat().st_size,'packed_bytes':b.stat().st_size,'original_base_sha256':sha(a),'packed_sha256':sha(b),'zip_entries_exact':True,'all_checkpoint_tensors_exact':True,'tensor_count':len(x)})
result={'reference':'original untouched frozen base12, not partially packed resume inputs','original_base_bytes':sum(r['original_base_bytes'] for r in rows),'packed_bytes':sum(r['packed_bytes'] for r in rows),'files':rows,'all_original_base_hashes_unchanged':True}
(ROOT/'canonical_lossless_storage_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
