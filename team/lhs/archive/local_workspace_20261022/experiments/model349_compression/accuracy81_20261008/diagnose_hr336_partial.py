"""Exploratory cached fusion using completed raw HR336 flip4, other sources unchanged."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
# Execute the already-audited cache loader/fusion extraction, without its search loop.
ns={'__file__':str(ROOT/'fusion_search.py')}
code=(ROOT/'fusion_search.py').read_text().split('GRID=list')[0];exec(compile(code,str(ROOT/'fusion_search.py'),'exec'),ns)
m=ns['m'];root=ns['ROOT'];combo=(12,3,13,1,1,1,0,1,7);ref=np.load(root.parent/'exhaustive_review_20261008/replay_iou.npz');rows=[];base=[];candidate=[];hashes={}
for i,name in enumerate(m.NAMES):
 z=m.load(name);old=m.predict(z,combo);gt=z['gt']
 p=root/'hr336_flip4_probabilities'/(name+'.npy');new=np.load(p);assert new.shape==z['336'].shape and new.dtype==np.float16 and np.isfinite(new).all()
 hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();z['336']=new.astype(np.float32);pred=m.predict(z,combo)
 def score(mask):return [np.count_nonzero((mask==c)&(gt==c))/u if (u:=np.count_nonzero((mask==c)|(gt==c))) else 0. for c in range(4)]
 a,b=np.array(score(old)),np.array(score(pred));assert np.array_equal(a,ref['best14'][i]);base.append(a);candidate.append(b)
 rows.append({'image':name,'base_cached_percent':float(a.mean()*100),'hr336_only_cached_percent':float(b.mean()*100),'delta_pp':float((b-a).mean()*100),'class_delta_pp':((b-a)*100).tolist()})
a,b=np.array(base),np.array(candidate);delta=(b-a).mean(1)*100
receipt=json.loads((root/'hr336_probability_hashes.json').read_text());assert hashes==receipt
r={'scope':'partial experiment: raw HR336 flip4 + existing cached identity sources; NOT full raw model inference / full HR336+347 flip4 / generalization proof','base_cached_percent':float(a.mean()*100),'candidate_cached_percent':float(b.mean()*100),'delta_pp':float(delta.mean()),'class_delta_pp':((b-a).mean(0)*100).tolist(),'worst_image':m.NAMES[int(delta.argmin())],'worst_delta_pp':float(delta.min()),'per_image':rows,'probability_hashes_verified_server_local':True,'probability_hashes':hashes,'other_sources_changed':False,'original_baseline_iou_exact':True,'independent_generalization_verified':False}
(root/'hr336_partial_diagnostic.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');np.savez(root/'hr336_partial_iou.npz',names=np.array(m.NAMES),base=a,candidate=b);print(json.dumps({k:v for k,v in r.items() if k not in ['per_image','probability_hashes']},ensure_ascii=False,indent=2))
