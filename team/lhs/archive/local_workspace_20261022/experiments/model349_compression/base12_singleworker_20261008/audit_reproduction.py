"""Post-completion read-only probability parity and complete reproduction hashes."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/model349_compress/base12_singleworker_20261008')
PREVIOUS=ROOT.parent/'base12_resident_packed_20261008'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (ROOT/'completion.json').exists()
 before=PREVIOUS/'runs/packed_fast12_valid20/probabilities';after=ROOT/'runs/singleworker12_valid20/probabilities';rows={}
 for component in ['326','329','336','269','base']:
  files=sorted((before/component).glob('*.pkl' if component=='base' else '*.npy'));assert len(files)==20;out=[]
  for p in files:
   q=after/component/p.name;assert q.exists()
   if component=='base':
    with p.open('rb') as f:a=pickle.load(f)
    with q.open('rb') as f:b=pickle.load(f)
    pairs={k:(v,b[k]) for k,v in a.items() if isinstance(v,np.ndarray)}
   else:pairs={'probability':(np.load(p),np.load(q))}
   fields={}
   for k,(a,b) in pairs.items():
    assert a.shape==b.shape and a.dtype==b.dtype
    fields[k]={'exact':bool(np.array_equal(a,b)),'different_values':int(np.count_nonzero(a!=b)),'max_absolute_difference':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),'dtype':str(a.dtype),'shape':list(a.shape)}
   out.append({'name':p.stem,'fields':fields})
  rows[component]=out
 write(ROOT/'probability_parity.json',{'reference':'preceding packed_fast12 raw valid; different run, numerical equality not assumed','rows':rows})
 for bundle in (ROOT/'bundles').iterdir():
  files=[p for p in bundle.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json'] and p.name!='reproduction_hashes.json' and '__pycache__' not in p.parts]
  write(bundle/'reproduction_hashes.json',{'files':{str(p.relative_to(bundle)):sha(p) for p in sorted(files)},'checkpoint_manifest_sha256':sha(bundle/'checkpoint_manifest.json')})
 print(json.dumps({c:{'exact_images':sum(all(f['exact'] for f in r['fields'].values()) for r in rs),'max_difference':max(f['max_absolute_difference'] for r in rs for f in r['fields'].values())} for c,rs in rows.items()},indent=2))
if __name__=='__main__':main()
