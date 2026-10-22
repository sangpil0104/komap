"""Recompute masks/IoUs from existing probabilities only; never raw-inference scores."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import ast,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[2]
SCRATCH=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad')
os.environ['SLIM']=str(SCRATCH/'slim');os.environ['GT']=str(SCRATCH/'gt/valid/masks')
sys.path.insert(0,str(WORK/'Model349_lite12_base/src'))
# Extract exactly the recorded constant, without importing unused GPU/training modules.
source=WORK/'Model349_reproduction_code/experiments/historical_scripts/highres1024_search.py'
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MATRIX' for t in node.targets):matrix=np.array(ast.literal_eval(node.value.args[0]),dtype=np.float32);break
stub=types.ModuleType('highres1024_search');stub.MATRIX=matrix;sys.modules['highres1024_search']=stub
os.chdir(WORK/'Model349_lite12_base')
spec=importlib.util.spec_from_file_location('cached',ROOT.parent/'exhaustive_20261008/exhaustive_local.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import cv2
cv2.setNumThreads(2)
configs={'base12':(0,5,15,1,0,1,0,0,7),'lite14':(0,15,15,1,0,1,0,0,7),'best14':(12,3,13,1,1,1,0,1,7),'best_no347':(12,3,13,1,0,1,0,1,7),'best_no322':(0,3,13,1,1,1,0,1,7),'best_326basefolds':(12,5,13,1,1,1,0,1,7),'best_329allfolds':(12,3,15,1,1,1,0,1,7),'best_no_pair_flag':(12,3,13,1,1,1,0,0,7)}
results={k:[] for k in configs};meta=[];t=time.monotonic()
for name in m.NAMES:
 z=m.load(name);meta.append({'name':name,'routed':z['meta']['routed'],'pair_weight':z['meta']['params']['pair_weight']})
 for key,combo in configs.items():
  pred=m.predict(z,combo);gt=z['gt'];ious=[]
  for cid in range(4):
   union=np.count_nonzero((pred==cid)|(gt==cid));ious.append(float(np.count_nonzero((pred==cid)&(gt==cid))/union) if union else 0.0)
  results[key].append(ious)
  if key in ['base12','best14']:
   out=ROOT/'cached_masks'/key;out.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out/(name+'.png')),pred)
 print(name,round(time.monotonic()-t,1),flush=True)
arrays={k:np.array(v) for k,v in results.items()};np.savez(ROOT/'replay_iou.npz',names=np.array(m.NAMES),**arrays)
summary={k:{'combo':configs[k],'miou_percent':float(v.mean()*100),'class_iou_percent':(v.mean(0)*100).tolist(),'vs_best14_pp':float((v-arrays['best14']).mean()*100),'vs_base12_pp':float((v-arrays['base12']).mean()*100),'worst_vs_base12_pp':float((v.mean(1)-arrays['base12'].mean(1)).min()*100)} for k,v in arrays.items()}
cache=np.load(SCRATCH/'slim'/f'{m.NAMES[0]}.npz')
summary['_provenance']={'kind':'cached probability replay, NOT raw-image inference','high_MATRIX_source':str(source.relative_to(WORK)),'high_MATRIX_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'slim_path':str(SCRATCH/'slim'),'q_dtype':str(cache['q'].dtype),'metadata':meta,'elapsed_seconds':time.monotonic()-t}
(ROOT/'replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
