import sys,json,pickle,copy,importlib.util,time
from pathlib import Path
import numpy as np
from PIL import Image
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('baseline_remaining',K/'codex_lite13_E_seed5_reduction_20261014/baseline_snapshot.py')
b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
assert abs(b.BASE-80.72098350464233)<1e-10
if sys.argv[1]=='preflight':
 (D/'baseline_preflight.json').write_text(json.dumps({'score_percent':b.BASE,'exact20':True,'coefficients_retuned':False}))
 sys.exit(0)
job=json.loads(Path(sys.argv[1]).read_text());tag=job['tag'];component=job['component']
folder=Path(job['work'])/'probabilities'/component;assert (folder/'complete.json').exists()
preds={}
for n,rec in b.recs.items():
 r=copy.copy(rec);sources=dict(rec['src'])
 if component=='base':
  with (folder/f'{n}.pkl').open('rb') as f:r['item']=pickle.load(f)
 else:sources[component]=np.load(folder/f'{n}.npy').astype(np.float32)
 preds[n]=b.finalize(r,b.predict_r(r,b.CFG,sources))
result=b.score(preds);result['score_percent']=b.BASE+result['delta'];result['family']=job['family'];result['seed']=job['seed'];result['tag']=tag
out=D/'scores';out.mkdir(exist_ok=True);(out/f'{tag}.json').write_text(json.dumps(result,indent=2))
p=D/'predictions'/tag;p.mkdir(parents=True,exist_ok=True)
for n,v in preds.items():Image.fromarray(v).save(p/f'{n}.png')
rows=[json.loads(p.read_text()) for p in sorted(out.glob('*.json'))];families={}
for fam in sorted(set(r['family'] for r in rows)):
 rs=[r for r in rows if r['family']==fam];vals=np.array([r['score_percent'] for r in rs])
 image_ranges={n:float(np.ptp([r['per_image'][n] for r in rs])) for n in b.names}
 families[fam]={'n':len(rs),'mean_percent':float(vals.mean()),'sample_sd_pp':float(vals.std(ddof=1)) if len(rs)>1 else None,'min_percent':float(vals.min()),'max_percent':float(vals.max()),'range_pp':float(np.ptp(vals)),'image_seed_ranges_pp':image_ranges,'rows':rs}
(D/'seed_variability.json').write_text(json.dumps({'baseline':b.BASE,'fixed_coefficients':True,'whole_pipeline_retraining':False,'families':families},indent=2))
print(tag,result['score_percent'],result['worst_d'],flush=True)
