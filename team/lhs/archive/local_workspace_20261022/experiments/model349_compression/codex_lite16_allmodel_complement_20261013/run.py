import os,sys,json,time,hashlib,traceback,importlib.util
from pathlib import Path
import numpy as np
S=Path('/workspace/model349_compress/codex_allmodel_fold_audit_20261013');O=Path('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013')
spec=importlib.util.spec_from_file_location('audit',S/'review.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=1)+'\n')
def summary(rows):
 ds=[r['delta_pp'] for r in rows.values()];ci=np.asarray([r['class_delta_pp'] for r in rows.values()]);return {'percent':80.55464328764197+float(np.mean(ds)),'delta_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_image_class_delta_pp':float(ci.min()),'mean_class_delta_pp':ci.mean(0).tolist(),'better':sum(x>1e-12 for x in ds),'worse':sum(x< -1e-12 for x in ds),'errors_fixed_pixels':sum(r['fixed'] for r in rows.values()),'new_errors_pixels':sum(r['new_errors'] for r in rows.values()),'pass_mean_local':bool(np.mean(ds)>=-1e-12 and min(ds)>=-.1)}
def task(c):
 import torch
 dest=O/f'results/cache{c["id"]:04d}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 oracle={};data={};digest=hashlib.sha256();fu=audit.GLOBAL['fu']
 for n,rec in audit.GLOBAL['records'].items():
  p=audit.read(c,n);assert p.shape==rec['e'].shape and np.isfinite(p).all() and p.min()>=-1e-6 and np.max(np.abs(p.sum(0)-1))<.04
  digest.update(p.tobytes());lab=fu.labels(p);base=rec['base'];g=rec['gt'];oldok=base==g;newok=lab==g
  oracle[n]={'baseline_error_pixels':int((~oldok).sum()),'candidate_correct_on_baseline_error':int((~oldok&newok).sum()),'candidate_wrong_on_baseline_correct':int((oldok&~newok).sum()),'both_wrong':int((~oldok&~newok).sum()),'both_wrong_different_class':int((~oldok&~newok&(base!=lab)).sum()),'recoverable_by_gt_class':[int((~oldok&newok&(g==i)).sum()) for i in range(4)],'candidate_wrong_by_gt_class':[int((oldok&~newok&(g==i)).sum()) for i in range(4)]}
  predictions={}
  for alpha in [.05,.1,.2]:
   e=(1-alpha)*rec['e']+alpha*p
   with torch.no_grad():v,_=audit.GLOBAL['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),rec['gray'])
   ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);predictions[f'E_blend_{alpha}']=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,rec['r'])).astype(np.uint8)
  top=np.partition(p,-2,axis=0)[-2:];gate=(rec['fl']==2)&(rec['ex']!=2)&(lab==rec['ex'])&(top[1]>=.9)&((top[1]-top[0])>=.4)
  predictions['F_Al_veto']=np.where(gate,rec['ex'],base).astype(np.uint8)
  bci,bmi=audit.iou(base,g)
  for tag,pred in predictions.items():
   ci,mi=audit.iou(pred,g);data.setdefault(tag,{})[n]={'miou':mi,'delta_pp':100*(mi-bmi),'class_delta_pp':(100*(np.array(ci)-bci)).tolist(),'fixed':int((~oldok&(pred==g)).sum()),'new_errors':int((oldok&(pred!=g)).sum()),'changed_pixels':int((pred!=base).sum())}
 totals={k:sum(v[k] for v in oracle.values()) for k in ['baseline_error_pixels','candidate_correct_on_baseline_error','candidate_wrong_on_baseline_correct','both_wrong','both_wrong_different_class']}
 totals['recoverable_by_gt_class']=np.sum([v['recoverable_by_gt_class'] for v in oracle.values()],axis=0).tolist();totals['candidate_wrong_by_gt_class']=np.sum([v['candidate_wrong_by_gt_class'] for v in oracle.values()],axis=0).tolist()
 result={'cache':c,'probability_sha256':digest.hexdigest(),'oracle_diagnostic_not_inference':totals,'oracle_per_image':oracle,'summary':{k:summary(v) for k,v in data.items()},'per_image':data,'valid_development_screen':True,'additional_checkpoint_count_pending_verification':True,'gt_used_in_rules':False,'fresh_inference':False};dump(dest/'result.json',result);return result
def main():
 if (O/'completion.json').exists():return
 cs=[c for c in json.loads((S/'coverage.json').read_text()) if c['complete20'] and c['channel_order']]
 resolved=json.loads((S/'resolved_channels.json').read_text());rejected=[]
 for c in resolved:
  proof=c['channel_proof'];path=Path(c['folder']);parent=path.parent if path.suffix else path
  if proof.startswith('recorded '):
   source=Path(proof.split(' ',2)[2]);allowed=[parent]+([parent.parent] if parent.parent.name not in ['outputs','komap_model1','model349_compress'] else [])
   if source.parent not in allowed:rejected.append({'id':c['id'],'reason':'class metadata belongs to unrelated model','proof':proof});continue
  cs.append(c)
 cs=list({c['id']:c for c in cs}.values());dump(O/'coverage.json',{'candidates':cs,'cache_count':len(cs),'unresolved_not_measured':len(json.loads((S/'unresolved_channels.json').read_text()))+len(rejected),'rejected_channel_proofs':rejected,'missing_probabilities_not_inferred':True})
 audit.init();results=[];seen={};duplicates=[];failures=[]
 for i,c in enumerate(cs):
  try:
   prior=S/f'results/cache{c["id"]:04d}/result.json';hint=json.loads(prior.read_text()).get('probability_sha256') if prior.exists() else None
   if hint and hint in seen:duplicates.append({'cache_id':c['id'],'same_as':seen[hint]});continue
   r=task(c);key=r['probability_sha256']
   if key in seen:duplicates.append({'cache_id':c['id'],'same_as':seen[key]})
   else:seen[key]=c['id'];results.append(r)
  except Exception:
   err={'cache_id':c['id'],'trace':traceback.format_exc()};failures.append(err);dump(O/f'failure_cache{c["id"]:04d}.json',err)
  finally:
   dump(O/'progress.json',{'visited':i+1,'total':len(cs),'unique_evaluated':len(results),'duplicates':len(duplicates),'errors':len(failures),'time':time.time(),'pid':os.getpid()});print(i+1,'/',len(cs),'unique',len(results),'errors',len(failures),flush=True)
 ranking=[]
 for r in results:
  for tag,v in r['summary'].items():ranking.append({'cache_id':r['cache']['id'],'source':r['cache']['folder'],'variant':r['cache']['variant'],'rule':tag,**v})
 ranking.sort(key=lambda x:-x['delta_pp']);dump(O/'ranking.json',ranking);dump(O/'duplicates.json',duplicates);dump(O/'completion.json',{'time':time.time(),'unique_evaluated':len(results),'errors':len(failures),'duplicates':len(duplicates),'release_replaced':False})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
