from pathlib import Path
import json,time,traceback,importlib.util,hashlib,copy,os
import numpy as np
O=Path('/workspace/model349_compress/codex_lite15_model_sharing_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013');F=Path('/workspace/model349_compress/codex_lite15_B4_HR_reduction_20261013')
s=importlib.util.spec_from_file_location('scan',S/'review.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def main():
 if (O/'completion.json').exists():return
 while not (O/'warm_phase_ready.json').exists():dump('status.json',{'stage':'waiting preceding warmstart/EMA/SWA and soup validation','time':time.time(),'pid':os.getpid()});time.sleep(20)
 import torch
 m.init(False);G=m.audit.GLOBAL;expected=json.load(open(F/'expected_mask_sha.json'))
 with np.load(F/'R326plain3456_fp16.npz') as z:
  for n,r in G['records'].items():r['sources']['326']=z[n].astype(np.float32);r['r']=G['infer'].predict_fixed(r['item'],r['sources'],G['cfg']);r['base']=m.finish(r,r['ex'],r['fl']);assert hashlib.sha256(r['base'].tobytes()).hexdigest()==expected[n]
 m.BASE=80.68255541753359;data={};info={}
 for n,r in G['records'].items():
  donors={k:r['sources'][k] for k in ['326','322','347','269','329']};donors['N3']=r['parts']['N3'];rp={};ef={}
  def elabels(tag,e):
   if tag not in ef:
    with torch.no_grad():v,_=G['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),r['gray'])
    ef[tag]=(G['fu'].labels(e),G['fu'].labels(v.numpy()[0]))
   return ef[tag]
  def save(tag,pred,count,detail):data.setdefault(tag,{})[n]=m.score(r,pred);info[tag]={'checkpoint_count_if_shared_once':count,**detail}
  for donor in ['326','322','347','269','N3']:
   src=dict(r['sources']);src['329']=m.roundtrip(donors[donor]);rr=G['infer'].predict_fixed(r['item'],src,G['cfg']);rp[donor]=rr;save('R329_share_'+donor,m.finish(r,r['ex'],r['fl'],rr),12,{'R329_group_shared_from':donor})
  cfg=copy.deepcopy(G['cfg']);cfg['route']['gamma329']=[0,0,0,0];rr=G['infer'].predict_fixed(r['item'],r['sources'],cfg);save('R329_role_removed',m.finish(r,r['ex'],r['fl'],rr),12,{'R329_removed_without_replacement':True})
  for donor in ['326','322','347','329','N3']:
   for target in ['pair',1,3]:
    if target=='pair':b=m.roundtrip(donors[donor]);count=13
    else:
     other=3 if target==1 else 1;b=m.roundtrip((r['b4folds'][other]+donors[donor])/2);count=14
    e=m.expert(r,r['parts'],b);ex,fl=elabels(f'EB_{donor}_{target}',e);save(f'EB4_share_{donor}_{target}',m.finish(r,ex,fl),count,{'E_B4_slot_shared_from':donor,'target':target,'partial_family_FP16_limit':target!='pair'})
  ids=['s1f4','s2f1','s2f4']
  for donor in ['326','322','329','347','269']:
   targets=[(x,) for x in ids]+[('s2f1','s2f4'),tuple(ids)] if donor in ['326','322','329'] else [('s2f1','s2f4'),tuple(ids)]
   for ts in targets:
    parts=dict(r['parts']);parts.update({x:donors[donor] for x in ts});e=m.expert(r,parts);ex,fl=elabels('HR_'+donor+'_'.join(ts),e);save('EHR_share_'+donor+'_'+'_'.join(ts),m.finish(r,ex,fl),15-len(ts),{'E_HR_slots_shared_from':donor,'targets':ts})
  for rd in ['326','322','347','269','N3']:
   for ed in ['326','322','347','329','N3']:
    if ed=='329':continue  # The original three-fold R329 donor would remain loaded, so not a 10-weight reduction.
    e=m.expert(r,r['parts'],m.roundtrip(donors[ed]));ex,fl=elabels(f'EB_{ed}_pair',e);save(f'joint_R329_{rd}_EB4_{ed}',m.finish(r,ex,fl,rp[rd]),10,{'R329_group_shared_from':rd,'E_B4_pair_shared_from':ed})
  dump('status.json',{'stage':'model sharing CPU validation','images':len(next(iter(data.values()))),'total':20,'configs':len(data),'time':time.time()});print(n,flush=True)
 ranking=[{'key':k,**info[k],**m.summarize(rows)} for k,rows in data.items()];ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('per_image.json',data);dump('completion.json',{'time':time.time(),'configs':len(ranking),'baseline_percent':m.BASE,'current15_control_exact20':True,'duplicate_previous_N3_HR_shares_omitted':True,'runtime_alias_implementation_required_for_deployment':True,'release_replaced':False});dump('sharing_ready.json',{'time':time.time(),'source':'CPU model-sharing validation complete'})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
