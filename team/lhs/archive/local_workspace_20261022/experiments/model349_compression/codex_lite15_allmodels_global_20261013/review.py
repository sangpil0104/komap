import os,sys,json,time,hashlib,traceback,pickle,gc,itertools,importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np
OLD=Path('/workspace/model349_compress/codex_allmodel_fold_audit_20261013');PREV=Path('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013');SRC=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013');O=Path('/workspace/model349_compress/codex_lite15_allmodels_global_20261013');F=SRC/'fresh';BASE=80.68255541753359;R326_NPZ=Path('/workspace/model349_compress/codex_lite15_B4_HR_reduction_20261013/R326plain3456_fp16.npz')
spec=importlib.util.spec_from_file_location('audit',OLD/'review.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=1)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def roundtrip(v):return v.astype(np.float16).astype(np.float32)
def init(load_rfolds=True):
 import torch,yaml
 from PIL import Image
 audit.init();G=audit.GLOBAL
 G['cfg']=yaml.safe_load((O/'selected_R_lite16_N3.yaml').read_text())
 receipt=json.loads((O/'baseline_receipt.json').read_text());assert receipt['baseline']=='lite15_N3_R326plain3456' and abs(receipt['valid_image_miou_percent']-BASE)<1e-10
 assert sha(R326_NPZ)==receipt['R326_probability_npz_sha256']
 with np.load(R326_NPZ,allow_pickle=False) as z:r326={n:z[n].astype(np.float32) for n in z.files}
 for n,rec in G['records'].items():
  # Build this baseline from the adopted single plain-326 slot, never the old
  # lite16 two-fold source. The recomputed R mask must match the fresh package.
  sources=dict(rec['sources']);sources['326']=roundtrip(r326[n]);rec['sources']=sources
  rec['r']=G['infer'].predict_fixed(rec['item'],sources,G['cfg'])
  assert np.array_equal(rec['r'],np.asarray(Image.open(O/f'baseline/R_fresh/predicted_labels/{n}.png'))),'lite15 R baseline mismatch '+n
  assert sha(O/f'baseline/R_fresh/predicted_labels/{n}.png')==receipt['R_labels_and_final_labels_sha256'][n]['R_label_sha256']
  load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)[[1,2,0,3]]
  rec['parts']={'N3':load('N3'),'s1f4':load('hr_s1_f4'),'s2f1':load('hr_s2_f1'),'s2f4':load('hr_s2_f4')};rec['b4']=load('b4ws');rec['e']=np.load(O/f'baseline/expert/{n}.npy').astype(np.float32);assert sha(O/f'baseline/expert/{n}.npy')==receipt['fresh_E_probability_sha256'][n];rec['base']=np.asarray(Image.open(O/f'baseline/final/predicted_labels/{n}.png'))
  assert sha(O/f'baseline/final/predicted_labels/{n}.png')==receipt['R_labels_and_final_labels_sha256'][n]['final_label_sha256']
  rec['rfolds']={(c,f):np.load(SRC/f'add/pr/valid/{c}_f{f}/{n}.npy').astype(np.float32) for c,ids in [('322',[3]),('326',[1,2]),('329',[1,3,4])] for f in ids} if load_rfolds else {}
  rec['b4folds']={f:np.load(audit.B/f'add/pe/valid/model329_ws_f{f}/{n}.npy').astype(np.float32)[[1,2,0,3]] for f in [1,3]}
  assert np.array_equal(expert(rec,rec['parts']),rec['e'])
  with torch.no_grad():v,_=G['refiner'](torch.from_numpy(np.log(np.maximum(rec['e'],1e-7))[None]),rec['gray'])
  rec['ex']=G['fu'].labels(rec['e']);rec['fl']=G['fu'].labels(v.numpy()[0]);assert np.array_equal(finish(rec,rec['ex'],rec['fl']),rec['base'])
  for k in ['hr','h1','h2']:rec.pop(k,None)
 assert abs(100*np.mean([audit.iou(r['base'],r['gt'])[1] for r in G['records'].values()])-BASE)<1e-10
 dump(O/'baseline_preflight.json',{'baseline':'lite15_N3_R326plain3456','baseline_percent':BASE,'images':len(G['records']),'R_fresh_exact20':True,'E_family_reconstructed_exact20':True,'final_mask_exact20':True,'R326_probability_sha256':sha(R326_NPZ),'fresh_E_probability_sha256':receipt['fresh_E_probability_sha256'],'input_sha256_unchanged':True,'test_GT_used':False})
 print('NEW_N3_BASELINE20_EXACT',flush=True)
def expert(rec,parts,b4=None):return .5*(rec['b4'] if b4 is None else b4)+.25*roundtrip((parts['N3']+parts['s1f4'])/2)+.25*roundtrip((parts['s2f1']+parts['s2f4'])/2)
def finish(rec,ex,fl,r=None):return np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,rec['r'] if r is None else r)).astype(np.uint8)
def infer_e(rec,e):
 import torch
 assert np.isfinite(e).all() and e.min()>=-1e-6
 with torch.no_grad():v,_=audit.GLOBAL['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),rec['gray'])
 return finish(rec,audit.GLOBAL['fu'].labels(e),audit.GLOBAL['fu'].labels(v.numpy()[0]))
def summarize(rows):
 ds=[r['delta_pp'] for r in rows.values()];ci=np.array([r['class_delta_pp'] for r in rows.values()]);return {'percent':BASE+float(np.mean(ds)),'delta_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_image':min(rows,key=lambda n:rows[n]['delta_pp']),'worst_class_delta_pp':float(ci.min()),'mean_class_delta_pp':ci.mean(0).tolist(),'better':sum(x>1e-12 for x in ds),'worse':sum(x< -1e-12 for x in ds),'fixed_pixels':sum(r['fixed_pixels'] for r in rows.values()),'new_error_pixels':sum(r['new_error_pixels'] for r in rows.values()),'pass_mean_local':bool(np.mean(ds)>=-1e-12 and min(ds)>=-.1)}
def score(rec,pred):
 bc,bm=audit.iou(rec['base'],rec['gt']);ci,mi=audit.iou(pred,rec['gt']);return {'miou':mi,'delta_pp':100*(mi-bm),'class_delta_pp':(100*(np.array(ci)-bc)).tolist(),'fixed_pixels':int(((rec['base']!=rec['gt'])&(pred==rec['gt'])).sum()),'new_error_pixels':int(((rec['base']==rec['gt'])&(pred!=rec['gt'])).sum())}
def task(c):
 dest=O/f'results/cache{c["id"]:04d}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 prior=PREV/f"results/cache{c['id']:04d}/result.json"
 if c['id']<10000 and prior.exists():c['expected_probability_sha256']=json.loads(prior.read_text())['probability_sha256']
 data={};oracle={};digest=hashlib.sha256();G=audit.GLOBAL;fu=G['fu']
 try:
  for n,rec in G['records'].items():
   p=audit.read(c,n);assert p.shape==rec['e'].shape and np.isfinite(p).all() and p.min()>=-1e-6 and np.max(np.abs(p.sum(0)-1))<.04;digest.update(p.tobytes());lab=fu.labels(p);base=rec['base'];gt=rec['gt'];oracle[n]={'candidate_correct_on_baseline_error':int(((base!=gt)&(lab==gt)).sum()),'candidate_wrong_on_baseline_correct':int(((base==gt)&(lab!=gt)).sum()),'recoverable_by_class':[int(((base!=gt)&(lab==gt)&(gt==i)).sum()) for i in range(4)]}
   def save(tag,pred):data.setdefault(tag,{})[n]=score(rec,pred)
   for alpha in [.05,.1,.2]:save(f'add_E_{alpha}',infer_e(rec,(1-alpha)*rec['e']+alpha*p))
   for tag in ['N3','s1f4','s2f1','s2f4']:
    parts=dict(rec['parts']);parts[tag]=roundtrip(p) if tag=='N3' else p;save('replace_E_'+tag,infer_e(rec,expert(rec,parts)))
   for f in [1,3]:
    b=roundtrip(np.maximum(rec['b4']+.5*(p-rec['b4folds'][f]),1e-7));b/=b.sum(0,keepdims=True);save(f'replace_E_B4f{f}',infer_e(rec,expert(rec,rec['parts'],b)))
   for tag,ids in [('E_HR3_to_one',['s1f4','s2f1','s2f4']),('E_HRseed2pair_to_one',['s2f1','s2f4'])]:
    parts=dict(rec['parts']);parts.update({x:p for x in ids});save(tag,infer_e(rec,expert(rec,parts)))
   save('E_B4pair_to_one',infer_e(rec,expert(rec,rec['parts'],roundtrip(p))))
   top=np.partition(p,-2,axis=0)[-2:];gate=(rec['fl']==2)&(rec['ex']!=2)&(lab==rec['ex'])&(top[1]>=.9)&((top[1]-top[0])>=.4);save('F_Al_veto',np.where(gate,rec['ex'],base).astype(np.uint8));save('replace_F_Al_decision',finish(rec,rec['ex'],lab))
   for component,ids in [('322',[3]),('329',[1,3,4])]:
    for f in ids:
     sources=dict(rec['sources']);v=np.maximum(sources[component]+(p-rec['rfolds'][component,f])/len(ids),1e-7);v=roundtrip(v/v.sum(0,keepdims=True));sources[component]=v;rr=G['infer'].predict_fixed(rec['item'],sources,G['cfg']);save(f'replace_R_{component}f{f}',finish(rec,rec['ex'],rec['fl'],rr))
    if len(ids)>1:
     sources=dict(rec['sources']);sources[component]=p;rr=G['infer'].predict_fixed(rec['item'],sources,G['cfg']);save(f'R_{component}group_to_one',finish(rec,rec['ex'],rec['fl'],rr))
   sources=dict(rec['sources']);sources['326']=p;rr=G['infer'].predict_fixed(rec['item'],sources,G['cfg']);save('R_326_current_slot_to_one',finish(rec,rec['ex'],rec['fl'],rr))
   for component in ['269','347']:
    sources=dict(rec['sources']);sources[component]=p;rr=G['infer'].predict_fixed(rec['item'],sources,G['cfg']);save('replace_R_'+component,finish(rec,rec['ex'],rec['fl'],rr))
   item=dict(rec['item']);item['preaux']=p;rr=G['infer'].predict_fixed(item,rec['sources'],G['cfg']);save('replace_R_base_distribution',finish(rec,rec['ex'],rec['fl'],rr))
  if c.get('expected_probability_sha256'):assert digest.hexdigest()==c['expected_probability_sha256'],'Previously verified candidate probabilities changed'
  result={'cache':c,'probability_sha256':digest.hexdigest(),'summary':{k:summarize(v) for k,v in data.items()},'per_image':data,'oracle_per_image_not_gate':oracle,'baseline_percent':BASE,'baseline':'lite15_N3_R326plain3456 fresh source output','valid_development_screen':True,'cached_R_B4_individual_precision_limit':True,'base_distribution_not_TENT_source_fresh':True,'F_Al_role_not_refiner_checkpoint_swap':True,'actual_weight_cost_pending_provenance':True};dump(dest/'result.json',result);return result
 except Exception:
  err={'cache':c,'error':traceback.format_exc()};dump(dest/'failure.json',err);return err
def focused():
 dest=O/'HR_removal.json'
 if dest.exists():return
 data={}
 for n,rec in audit.GLOBAL['records'].items():
  ids=['s1f4','s2f1','s2f4']
  for size in [0,1,2]:
   for keep in itertools.combinations(ids,size):
    # Preserve .25/.25 family coefficients when both remain; redistribute HR family budget to group1 if seed2 is empty.
    h1=roundtrip(np.mean([rec['parts']['N3']]+([rec['parts']['s1f4']] if 's1f4' in keep else []),axis=0));h2keys=[x for x in ['s2f1','s2f4'] if x in keep]
    if h2keys:e=.5*rec['b4']+.25*h1+.25*roundtrip(np.mean([rec['parts'][x] for x in h2keys],axis=0))
    else:e=.5*rec['b4']+.5*h1
    tag='keep_'+('_'.join(keep) if keep else 'none');data.setdefault(tag,{})[n]=score(rec,infer_e(rec,e))
  for replace in [('s1f4',),('s2f1',),('s2f4',),('s2f1','s2f4'),tuple(ids)]:
   parts=dict(rec['parts']);parts.update({x:rec['parts']['N3'] for x in replace});tag='share_N3_'+('_'.join(replace));data.setdefault(tag,{})[n]=score(rec,infer_e(rec,expert(rec,parts)))
  # N3 and an existing R member can fill HR positions without adding any checkpoint.
  for donor in ['347','269']:
   for target in ids:
    parts=dict(rec['parts']);parts[target]=rec['sources'][donor];tag=f'share_R{donor}_{target}';data.setdefault(tag,{})[n]=score(rec,infer_e(rec,expert(rec,parts)))
 summary={k:summarize(v) for k,v in data.items()};dump(dest,{'baseline_percent':BASE,'summary':summary,'per_image':data,'automatic_adoption':False,'HR3_remaining':True,'family_reweight_if_seed2_removed':True,'fixed_coefficients_otherwise':True});print('HR_FOCUSED_COMPLETE',len(summary),flush=True)
def candidates():
 cover=json.loads((PREV/'coverage.json').read_text());dup={r['cache_id'] for r in json.loads((PREV/'duplicates.json').read_text())};cs=[c for c in cover['candidates'] if c['id'] not in dup]
 for c in cs:
  prior=PREV/f"results/cache{c['id']:04d}/result.json"
  if prior.exists():c['expected_probability_sha256']=json.loads(prior.read_text())['probability_sha256']
 names=list(cs[0]['files'])
 for tag,seeds in [('N3',[414,415,416,417]),('L2',[361,362,363,364])]:
  for s in seeds:
   folder=SRC/f'codex_{"N3" if tag=="N3" else "L2"}_seed4_20261013/evaluation/seed{s}/probabilities';cs.append({'id':10000+len(cs),'folder':str(folder),'storage':'npy','variant':tag+str(s),'channel_order':'main','channel_proof':'verified fresh evaluation TEAM_TO_MAIN conversion','complete20':True,'files':{n:{'path':str(folder/f'{n}.npy')} for n in names},'weight_cost':1,'weight_cost_proof':'one newly trained seed checkpoint'})
 return cs
def main():
 if (O/'completion.json').exists():return
 assert (O/'fresh_transfer_ready.json').exists(),'lite15 fresh baseline transfer not verified'
 dump(O/'status.json',{'stage':'lite15 baseline preflight','time':time.time(),'pid':os.getpid()})
 init(False);dump(O/'status.json',{'stage':'lite15 HR reduction focused checks','time':time.time(),'pid':os.getpid()});focused();audit.GLOBAL.clear();gc.collect()
 assert (SRC/'transfer_complete.json').exists(),'Existing candidate probability transfer incomplete'
 dump(O/'status.json',{'stage':'lite15 full global candidate scan','time':time.time(),'pid':os.getpid()})
 cs=candidates();dump(O/'coverage.json',{'eligible_unique_prior_candidates':294,'new_seed_candidates':8,'candidates':cs,'unverified_previous_complete_caches':112,'checkpoint_without_cache_not_inferred':True});results=[]
 # P100 host has 48 CPU cores; five workers stay within a modest CPU/RAM
 # footprint while reducing the long per-candidate CPU-only screening time.
 with get_context('spawn').Pool(5,initializer=init) as pool:
  for r in pool.imap_unordered(task,cs):
   results.append(r);dump(O/'progress.json',{'completed':len(results),'total':len(cs),'last_cache':r['cache']['id'],'errors':sum('error' in q for q in results),'pid':os.getpid(),'time':time.time()});print(len(results),'/',len(cs),'error' if 'error' in r else 'ok',flush=True)
 rankings=[]
 for r in results:
  if 'error' in r:continue
  for rule,v in r['summary'].items():rankings.append({'source':r['cache']['folder'],'variant':r['cache']['variant'],'cache_id':r['cache']['id'],'rule':rule,**v})
 rankings.sort(key=lambda x:-x['delta_pp']);dump(O/'ranking.json',rankings);dump(O/'completion.json',{'time':time.time(),'candidates':len(cs),'errors':sum('error' in r for r in results),'rules_per_candidate':len(next(r for r in results if 'error' not in r)['summary']),'baseline_percent':BASE,'release_replaced':False})
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
