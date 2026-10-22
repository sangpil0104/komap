from pathlib import Path
import json,hashlib,time,traceback,numpy as np
O=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
try:
 while not (O/'transfer_complete.json').exists():
  source=O/'source_probability_sha.json'
  try:d=json.loads(source.read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
  ready=True
  for rel in d:
   p=O/rel
   if not p.exists():ready=False;break
   if p.suffix=='.npy':
    try:a=np.load(p,mmap_mode='r');del a
    except Exception:ready=False;break
  if not ready:time.sleep(10);continue
  mismatches=[]
  for rel,h in d.items():
   v=hashlib.sha256();p=O/rel
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8388608),b''):v.update(b)
   if v.hexdigest()!=h:mismatches.append(rel)
  if mismatches:raise RuntimeError('transfer SHA mismatch '+str(mismatches[:5]))
  (O/'transfer_complete.json').write_text(json.dumps({'time':time.time(),'source_probabilities_SHA_exact':True,'files':len(d)}));print('TRANSFER_SHA_EXACT',len(d),flush=True)
except Exception:(O/'transfer_failure.json').write_text(json.dumps({'trace':traceback.format_exc()}));raise
