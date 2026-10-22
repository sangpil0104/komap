import os,json,time,hashlib,traceback,importlib.util,gc
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_B4R_newEF_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
s=importlib.util.spec_from_file_location('screen',S/'review.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
s=importlib.util.spec_from_file_location('newF',O/'train_F.py');nf=importlib.util.module_from_spec(s);s.loader.exec_module(nf)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def label(p):return m.audit.GLOBAL['fu'].labels(p)
def Eprob(rec,p,rule):
 if rule=='standalone':return p
 if rule=='add25':return .75*rec['e']+.25*p
 if rule=='B4pair':return m.expert(rec,rec['parts'],m.roundtrip(p))
 parts=dict(rec['parts'])
 if rule=='HR3':parts.update({k:p for k in ['s1f4','s2f1','s2f4']})
 else:parts[rule]=m.roundtrip(p) if rule=='N3' else p
 return m.expert(rec,parts)
def main():
 if (O/'completion.json').exists():return
 import torch,cv2
 torch.set_num_threads(2);cv2.setNumThreads(2);m.init(False);G=m.audit.GLOBAL
 rc=json.load(open('/workspace/model349_compress/codex_R_start_ablation_N3_20261013/reopt_result.json'))['best']['B4_4fold']['config'];cs=m.candidates();byid={c['id']:c for c in cs};basec=byid[146]
 for n,r in G['records'].items():
  r['B4base']=m.audit.read(basec,n);it=dict(r['item']);it['preaux']=r['B4base'];r['r_new']=G['infer'].predict_fixed(it,r['sources'],rc)
 assert abs(100*np.mean([m.audit.iou(r['r_new'],r['gt'])[1] for r in G['records'].values()])-79.83059410105437)<1e-9
 basefinal=np.mean([m.audit.iou(m.finish(r,r['ex'],r['fl'],r['r_new']),r['gt'])[1] for r in G['records'].values()])*100;assert abs(basefinal-80.58499736520488)<1e-9
 chosen=[];keys=set()
 def add(c):
  key=c.get('expected_probability_sha256') or str(c['id'])
  if key not in keys:keys.add(key);chosen.append(c)
 for prior in ['codex_lite16_allmodel_complement_20261013','codex_allmodel_fold_audit_20261013']:
  count=0
  for x in json.load(open('/workspace/model349_compress/'+prior+'/ranking.json')):
   if x['cache_id'] in byid:
    prev=len(chosen);add(byid[x['cache_id']]);count+=len(chosen)>prev
    if count>=12:break
 for c in cs:
  if c['id']>=10000 or c['id'] in [146,301,302,303,304,305,306]:add(c)
 dump('candidate_coverage.json',{'candidates':chosen,'count':len(chosen),'not_full302_scan':True,'sources_preselected_using_prior_valid_screen':True,'R_percent':79.83059410105437,'R_unchanged':True,'oldEF_baseline_percent':basefinal,'current_ResNet_REF_percent':m.BASE})
 rows={};plans={};checks={}
 def save(tag,rec,pred,plan):rows.setdefault(tag,{})[rec['name']]=m.score(rec,pred);plans[tag]=plan
 for n,r in G['records'].items():r['name']=n
 for n,r in G['records'].items():
  for bits in [5,15]:save(f'control_{bits}',r,np.where(np.isin(r['ex'],[i for i in range(4) if bits&(1<<i)]),r['ex'],r['r_new']),{'candidate_id':None,'rule':'current_E','class_bits':bits})
 for idx,c in enumerate(chosen):
  digest=hashlib.sha256()
  for n,r in G['records'].items():
   p=m.audit.read(c,n);assert p.shape==r['e'].shape and np.isfinite(p).all() and p.min()>=-1e-6;digest.update(p.tobytes())
   for rule in ['standalone','add25','B4pair','HR3','N3','s1f4','s2f1','s2f4']:
    ex=label(Eprob(r,p,rule))
    for bits in [5,15]:
     pred=np.where(np.isin(ex,[i for i in range(4) if bits&(1<<i)]),ex,r['r_new']);tag=f'c{c["id"]}_{rule}_{bits}';save(tag,r,pred,{'candidate_id':c['id'],'rule':rule,'class_bits':bits})
  checks[str(c['id'])]=digest.hexdigest()
  if c.get('expected_probability_sha256'):assert digest.hexdigest()==c['expected_probability_sha256']
  dump('progress.json',{'stage':'new E member substitutions/additions','completed_candidates':idx+1,'total':len(chosen),'configs':len(rows),'pid':os.getpid(),'time':time.time()});dump('E_coarse_summary.json',{k:m.summarize(v) for k,v in rows.items()})
 ranked=sorted(rows,key=lambda k:-m.summarize(rows[k])['percent']);local=sorted(rows,key=lambda k:-m.summarize(rows[k])['worst_image_delta_pp']);skeletons=[];used=set()
 for key in ranked[:4]+local[:4]+['control_5']:
  plan=plans[key];sig=(plan['candidate_id'],plan['rule'])
  if sig not in used:used.add(sig);skeletons.append(plan)
 refined={};rp={}
 for plan in skeletons:
  for n,r in G['records'].items():
   e=r['e'] if plan['candidate_id'] is None else Eprob(r,m.audit.read(byid[plan['candidate_id']],n),plan['rule']);ex=label(e)
   for bits in range(16):
    tag=f'c{plan["candidate_id"]}_{plan["rule"]}_{bits}';refined.setdefault(tag,{})[n]=m.score(r,np.where(np.isin(ex,[i for i in range(4) if bits&(1<<i)]),ex,r['r_new']));rp[tag]={**plan,'class_bits':bits}
 dump('E_refined_summary.json',{k:{'plan':rp[k],**m.summarize(v)} for k,v in refined.items()});dump('E_source_probability_sha256.json',checks)
 best=sorted(refined,key=lambda k:-m.summarize(refined[k])['percent']);safe=sorted(refined,key=lambda k:-m.summarize(refined[k])['worst_image_delta_pp']);selected=list(dict.fromkeys(best[:3]+safe[:2]+[next(k for k in rp if rp[k]['candidate_id'] is None and rp[k]['class_bits']==5)]));dump('selected_E_plans.json',{k:rp[k] for k in selected})
 while not (O/'F_training_completion.json').exists():
  if (O/'F_training_failure.json').exists():raise RuntimeError('new F training failed')
  dump('progress.json',{'stage':'waiting new F OOF training','E_candidates':len(chosen),'E_refined_configs':len(refined),'time':time.time()});time.sleep(10)
 models={}
 for tag in ['plain','balanced']:
  model=torch.nn.Linear(12,4).eval();model.load_state_dict(torch.load(O/f'F_{tag}.pt',map_location='cpu',weights_only=False)['state_dict']);models[tag]=model
 finalrows={};finalplans={}
 from PIL import Image
 for n,r in G['records'].items():
  raw=np.array(Image.open(m.audit.B/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255;feat=nf.features(r['B4base'],raw);tensor=torch.from_numpy(feat.reshape(12,-1).T.copy());fp={}
  with torch.no_grad():
   for tag,model in models.items():
    batches=[torch.softmax(model(x),1).numpy() for x in tensor.split(65536)];fp[tag]=np.concatenate(batches).T.reshape(r['e'].shape)
  for ek in selected:
   plan=rp[ek];e=r['e'] if plan['candidate_id'] is None else Eprob(r,m.audit.read(byid[plan['candidate_id']],n),plan['rule']);ex=label(e);cs=[i for i in range(4) if plan['class_bits']&(1<<i)];stage=np.where(np.isin(ex,cs),ex,r['r_new']);key=ek+'_noF';finalrows.setdefault(key,{})[n]=m.score(r,stage);finalplans[key]={'E':plan,'F':None}
   for ftag,p in fp.items():
    fl=label(p);conf=p.max(0)
    for threshold in [.6,.8,.9]:
     for disagree in [False,True]:
      for fb in [4,5,8,15]:
       gate=(conf>=threshold)&np.isin(fl,[i for i in range(4) if fb&(1<<i)])
       if disagree:gate&=(r['r_new']!=ex)
       pred=np.where(gate,fl,stage);key=f'{ek}_F{ftag}_t{threshold}_d{int(disagree)}_b{fb}';finalrows.setdefault(key,{})[n]=m.score(r,pred);finalplans[key]={'E':plan,'F':{'variant':ftag,'threshold':threshold,'only_R_E_disagreement':disagree,'class_bits':fb}}
  dump('progress.json',{'stage':'new E plus newly trained F','images':len(next(iter(finalrows.values()))),'total':20,'configs':len(finalrows),'time':time.time()});print('NEW_EF',n,flush=True)
 ranking=[{'key':k,'plan':finalplans[k],**m.summarize(v)} for k,v in finalrows.items()];ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('per_image.json',finalrows);dump('completion.json',{'time':time.time(),'E_candidate_members':len(chosen),'E_coarse_configs':len(rows),'E_refined_configs':len(refined),'new_F_models_trained':2,'final_configs':len(ranking),'best':ranking[0],'oldEF_baseline_percent':basefinal,'current_REF_percent':m.BASE,'new_R_percent':79.83059410105437,'new_F_train_OOF_only':True,'valid_repeated_development_selection':True,'candidate_checkpoint_cost_pending_provenance':True,'release_replaced':False});print('COMPLETE',json.dumps(ranking[:3]),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
