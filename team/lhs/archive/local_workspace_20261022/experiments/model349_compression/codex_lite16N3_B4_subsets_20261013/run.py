import sys,json,time,traceback,itertools,importlib.util
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_lite16N3_B4_subsets_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
spec=importlib.util.spec_from_file_location('newscan',S/'review.py');scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=1)+'\n')
def subsets(ids):return [t for n in range(1,len(ids)+1) for t in itertools.combinations(ids,n)]
def main():
 if (O/'completion.json').exists():return
 while not (S/'transfer_complete.json').exists():
  if (S/'transfer_failure.json').exists():raise RuntimeError('probability transfer verification failed')
  dump(O/'status.json',{'stage':'waiting verified probability transfer','time':time.time()});time.sleep(10)
 scan.init();G=scan.audit.GLOBAL;cfgs=list(itertools.product(subsets([1,2]),subsets([1,3,4]),subsets([1,3])));data={};info={}
 for n,rec in G['records'].items():
  rpred={};epred={}
  for f326,f329 in itertools.product(subsets([1,2]),subsets([1,3,4])):
   src=dict(rec['sources'])
   if len(f326)<2:src['326']=scan.roundtrip(np.mean([rec['rfolds']['326',f] for f in f326],axis=0))
   if len(f329)<3:src['329']=scan.roundtrip(np.mean([rec['rfolds']['329',f] for f in f329],axis=0))
   rpred[f326,f329]=G['infer'].predict_fixed(rec['item'],src,G['cfg'])
  for fb in subsets([1,3]):
   b=rec['b4'] if len(fb)==2 else scan.roundtrip(np.mean([rec['b4folds'][f] for f in fb],axis=0));e=scan.expert(rec,rec['parts'],b)
   import torch
   with torch.no_grad():v,_=G['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),rec['gray'])
   epred[fb]=(G['fu'].labels(e),G['fu'].labels(v.numpy()[0]))
  for f326,f329,fb in cfgs:
   tag=f"R326_{''.join(map(str,f326))}__R329_{''.join(map(str,f329))}__EB4_{''.join(map(str,fb))}";ex,fl=epred[fb];pred=scan.finish(rec,ex,fl,rpred[f326,f329]);data.setdefault(tag,{})[n]=scan.score(rec,pred);info[tag]={'R326_folds':list(f326),'R329_folds':list(f329),'E_B4ws_folds':list(fb),'checkpoint_count':16-(2-len(f326))-(3-len(f329))-(2-len(fb))}
   if (f326,f329,fb)==((1,2),(1,3,4),(1,3)):assert np.array_equal(pred,rec['base']),'full subset control mismatch'
  dump(O/'progress.json',{'completed_images':len(next(iter(data.values()))),'total_images':20,'configurations':63,'time':time.time()});print(n,flush=True)
 ranking=[{'key':k,**info[k],**scan.summarize(v)} for k,v in data.items()];ranking.sort(key=lambda x:(x['checkpoint_count'],-x['delta_pp']));dump(O/'ranking.json',ranking);dump(O/'per_image.json',data);dump(O/'completion.json',{'time':time.time(),'configurations':63,'baseline_percent':scan.BASE,'control_masks_exact':True,'fold_cache_precision_limit':True,'fresh_partial_subset_inference':False,'release_replaced':False})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
