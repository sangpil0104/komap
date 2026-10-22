import os,sys,json,time,hashlib,traceback,gc,importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
O=Path('/workspace/model349_compress/codex_lite14_top_candidates_20261013');OLD=O.parent/'codex_lite15_allmodels_global_20261013';BASE=80.72075530452783
spec=importlib.util.spec_from_file_location('lite15audit',OLD/'review.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def dump(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=1)+'\n');tmp.replace(p)
# Keep the completed lite15 experiment receipts untouched.
v.dump=lambda path,data:dump(O/f'lite15_control_{os.getpid()}_{path.name}',data)
def init():
 import torch
 from PIL import Image
 v.init();G=v.audit.GLOBAL;d=json.load(open(O/'baseline/completion.json'));expected={Path(p).name:h for p,h in d['input_sha256'].items() if '/valid/images/' in p};reconstruction={}
 for n,r in G['records'].items():
  assert v.sha(v.audit.B/f'data/valid/images/{n}_image.png')==expected[n+'_image.png']
  h70=np.load(O/f'baseline/H70/{n}.npy').astype(np.float32)[[1,2,0,3]];r['parts']={'N3':r['parts']['N3'],'s1f4':r['parts']['s1f4'],'H70':h70}
  r['e']=np.load(O/f'baseline/expert/{n}.npy').astype(np.float32);r['base']=np.asarray(Image.open(O/f'baseline/final/{n}.png'))
  r['reconstructed_e']=formula(r,r['parts']);err=float(np.abs(r['reconstructed_e']-r['e']).max());assert err<.0002,(n,err);reconstruction[n]=err
  with torch.no_grad():a,_=G['refiner'](torch.from_numpy(np.log(np.maximum(r['e'],1e-7))[None]),r['gray'])
  r['ex']=G['fu'].labels(r['e']);r['fl']=G['fu'].labels(a.numpy()[0]);assert np.array_equal(v.finish(r,r['ex'],r['fl']),r['base']),n
 assert abs(100*np.mean([v.audit.iou(r['base'],r['gt'])[1] for r in G['records'].values()])-BASE)<1e-9
 v.BASE=BASE
 dump(O/f'preflight_{os.getpid()}.json',dict(baseline_percent=BASE,baseline_mask_exact20=True,input_sha_exact20=True,E_probability_reconstruction_max_abs=max(reconstruction.values()),E_reconstruction_per_image=reconstruction,source_E_probability_hash={n:v.sha(O/f'baseline/expert/{n}.npy') for n in G['records']},test_GT_used=False))
 print('LITE14_EXACT20',os.getpid(),max(reconstruction.values()),flush=True)
def formula(r,parts):return .5*r['b4']+.25*v.roundtrip((parts['N3']+parts['s1f4'])/2)+.25*v.roundtrip(parts['H70'])
def expert(r,parts):
 # Anchor unchanged contributions to the verified fresh source E tensor.
 return r['e']+(formula(r,parts)-r['reconstructed_e'])
def predict(r,rule,p):
 G=v.audit.GLOBAL
 if rule.startswith('add_E_'):return v.infer_e(r,(1-float(rule[6:]))*r['e']+float(rule[6:])*p)
 if rule in ['replace_E_H70','blend_E_H70_0.5','share_E_HR2']:
  parts=dict(r['parts']);parts['H70']=(r['parts']['H70']+p)/2 if rule=='blend_E_H70_0.5' else p
  if rule=='share_E_HR2':parts['s1f4']=p
  return v.infer_e(r,expert(r,parts))
 if rule=='F_Al_veto':
  lab=G['fu'].labels(p);top=np.partition(p,-2,axis=0)[-2:];gate=(r['fl']==2)&(r['ex']!=2)&(lab==r['ex'])&(top[1]>=.9)&((top[1]-top[0])>=.4)
  return np.where(gate,r['ex'],r['base']).astype(np.uint8)
 component={'replace_R_269':'269','replace_R_322f3':'322','R_329group_to_one':'329'}[rule];src=dict(r['sources']);src[component]=p
 rr=G['infer'].predict_fixed(r['item'],src,G['cfg']);return v.finish(r,r['ex'],r['fl'],rr)
def task(entry):
 from PIL import Image
 c=entry['candidate'];dest=O/f'results/cache{c["id"]:04d}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.load(open(dest/'result.json'))
 try:
  rows={};digest=hashlib.sha256()
  for n,r in v.audit.GLOBAL['records'].items():
   p=v.audit.read(c,n);assert p.shape==r['e'].shape and np.isfinite(p).all() and p.min()>=-1e-6 and np.max(np.abs(p.sum(0)-1))<.04;digest.update(p.tobytes())
   for rule in entry['rules']:
    pred=predict(r,rule,p);rows.setdefault(rule,{})[n]=v.score(r,pred);out=dest/rule;out.mkdir(exist_ok=True);Image.fromarray(pred).save(out/f'{n}.png')
  assert digest.hexdigest()==c['expected_probability_sha256'],'source probabilities changed'
  result=dict(cache=c,rules=entry['rules'],selection_evidence=entry['selection_evidence'],probability_sha256=digest.hexdigest(),baseline_percent=BASE,baseline='lite14_N3_R326plain3456_HRtrain70seed5',summary={k:v.summarize(a) for k,a in rows.items()},per_image=rows,actual_weight_count_pending_provenance=True,valid_development_screen=True,test_GT_used=False,release_replaced=False)
  dump(dest/'result.json',result);return result
 except Exception:
  result=dict(cache=c,error=traceback.format_exc());dump(dest/'failure.json',result);return result

def direct_control():
 from PIL import Image
 rows={};digest=hashlib.sha256()
 for n,r in v.audit.GLOBAL['records'].items():
  p=r['sources']['347'];digest.update(p.tobytes());src=dict(r['sources']);src['269']=p;rr=v.audit.GLOBAL['infer'].predict_fixed(r['item'],src,v.audit.GLOBAL['cfg']);pred=v.finish(r,r['ex'],r['fl'],rr);rows[n]=v.score(r,pred);out=O/'direct_R347_to_R269';out.mkdir(exist_ok=True);Image.fromarray(pred).save(out/f'{n}.png')
 result=dict(rule='share_existing_R347_into_R269',checkpoint_count=13,actual_weight_count_verified=True,no_new_checkpoint=True,no_new_GPU_inference=True,source_probability_sha256=digest.hexdigest(),summary=v.summarize(rows),per_image=rows,baseline_percent=BASE)
 dump(O/'direct_control.json',result);return result

def main():
 if (O/'completion.json').exists():return
 assert (O/'transfer_ready.json').exists()
 shortlist=json.load(open(O/'shortlist.json'));entries=shortlist['candidates'];assert len(entries)==15
 dump(O/'status.json',dict(stage='lite14 preflight and shared R347 control',pid=os.getpid(),time=time.time()))
 init();direct=direct_control();v.audit.GLOBAL.clear();gc.collect();results=[]
 dump(O/'status.json',dict(stage='lite14 shortlisted CPU re-evaluation',pid=os.getpid(),time=time.time()))
 with get_context('spawn').Pool(4,initializer=init) as pool:
  for result in pool.imap_unordered(task,entries):
   results.append(result);dump(O/'progress.json',dict(completed_candidates=len(results),total_candidates=len(entries),completed_cases=sum(len(x.get('summary',{})) for x in results)+1,total_cases=shortlist['case_count'],errors=sum('error' in x for x in results),last_cache=result['cache']['id'],time=time.time()));print('DONE',len(results),'/',len(entries),flush=True)
 ranked=[]
 for result in results:
  if 'error' in result:continue
  for rule,row in result['summary'].items():ranked.append(dict(cache_id=result['cache']['id'],source=result['cache']['folder'],variant=result['cache']['variant'],rule=rule,actual_checkpoint_count=None,**row))
 ranked.append(dict(cache_id=None,rule=direct['rule'],actual_checkpoint_count=13,**direct['summary']));ranked.sort(key=lambda x:-x['delta_pp']);dump(O/'ranking.json',ranked)
 dump(O/'completion.json',dict(candidate_count=len(entries),case_count=len(ranked),errors=sum('error' in x for x in results),baseline_percent=BASE,CPU_only=True,test_GT_used=False,release_replaced=False,time=time.time()));dump(O/'status.json',dict(stage='complete',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',dict(trace=traceback.format_exc(),time=time.time()));raise
