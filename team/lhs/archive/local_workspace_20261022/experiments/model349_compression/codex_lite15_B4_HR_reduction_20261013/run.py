import json,time,traceback,importlib.util,hashlib,itertools,os
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_lite15_B4_HR_reduction_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
s=importlib.util.spec_from_file_location('scan',S/'review.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def subsets(xs):return [c for size in range(len(xs)+1) for c in itertools.combinations(xs,size)]
def main():
 if (O/'completion.json').exists():return
 m.init(True);G=m.audit.GLOBAL;expected=json.load(open(O/'expected_mask_sha.json'));data={};info={};source_errors=[]
 import torch
 with np.load(O/'R326plain3456_fp16.npz') as z:
  assert set(z.files)==set(G['records'])
  for n,r in G['records'].items():
   for component,ids in [('322',[3]),('326',[1,2]),('329',[1,3,4])]:
    averaged=m.roundtrip(np.mean([r['rfolds'][component,f] for f in ids],0));err=float(np.max(np.abs(averaged-r['sources'][component])));source_errors.append({'image':n,'component':component,'max_abs_error':err});assert err<.005,'per-fold MAIN-order average fails source comparison'
   r['sources']['326']=z[n].astype(np.float32);r['r']=G['infer'].predict_fixed(r['item'],r['sources'],G['cfg']);r['base']=m.finish(r,r['ex'],r['fl']);assert hashlib.sha256(r['base'].tobytes()).hexdigest()==expected[n],'current15 mask mismatch'
 m.BASE=80.68255541753359
 assert abs(100*np.mean([m.audit.iou(r['base'],r['gt'])[1] for r in G['records'].values()])-m.BASE)<1e-9
 dump('preflight.json',{'current15_masks_exact20':True,'R_fold_channel_order':'MAIN producer; no repeated permutation','source_average_checks':source_errors,'baseline_percent':m.BASE});print('CURRENT_LITE15_MASK20_EXACT',flush=True)
 folds=[x for x in subsets([1,3,4]) if x];ids=['s1f4','s2f1','s2f4'];keeps=subsets(ids)
 def ekeep(r,keep):
  h1=m.roundtrip(np.mean([r['parts']['N3']]+([r['parts']['s1f4']] if 's1f4' in keep else []),0));h2=[r['parts'][x] for x in ['s2f1','s2f4'] if x in keep]
  return .5*r['b4']+.25*h1+.25*m.roundtrip(np.mean(h2,0)) if h2 else .5*r['b4']+.5*h1
 for n,r in G['records'].items():
  rpred={}
  for fs in folds:
   src=dict(r['sources']);src['329']=r['sources']['329'] if len(fs)==3 else m.roundtrip(np.mean([r['rfolds']['329',f] for f in fs],0));rpred[fs]=G['infer'].predict_fixed(r['item'],src,G['cfg'])
  def final(e,rr):
   with torch.no_grad():v,_=G['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),r['gray'])
   return m.finish(r,G['fu'].labels(e),G['fu'].labels(v.numpy()[0]),rr)
  for keep in keeps:
   e=ekeep(r,keep)
   if len(keep)==3:assert np.array_equal(e,r['e'])
   with torch.no_grad():v,_=G['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),r['gray'])
   ex=G['fu'].labels(e);fl=G['fu'].labels(v.numpy()[0])
   for fs in folds:
    tag='R329_'+''.join(map(str,fs))+'__HR_'+('_'.join(keep) if keep else 'none');pred=m.finish(r,ex,fl,rpred[fs]);data.setdefault(tag,{})[n]=m.score(r,pred);info[tag]={'R329_folds':fs,'HR_folds_kept':keep,'checkpoint_count':9+len(fs)+len(keep)}
    if len(fs)==3 and len(keep)==3:assert np.array_equal(pred,r['base'])
  for targets in [('s1f4',),('s2f1',),('s2f4',),('s2f1','s2f4'),tuple(ids)]:
   p=dict(r['parts']);p.update({t:r['parts']['N3'] for t in targets});tag='share_N3_'+'_'.join(targets);data.setdefault(tag,{})[n]=m.score(r,final(m.expert(r,p),r['r']));info[tag]={'R329_folds':[1,3,4],'share_existing':'N3','targets':targets,'checkpoint_count':15-len(targets)}
  for donor in ['347','269']:
   for target in ids:
    p=dict(r['parts']);p[target]=r['sources'][donor];tag=f'share_R{donor}_{target}';data.setdefault(tag,{})[n]=m.score(r,final(m.expert(r,p),r['r']));info[tag]={'R329_folds':[1,3,4],'share_existing':donor,'targets':[target],'checkpoint_count':14}
  dump('progress.json',{'images':len(next(iter(data.values()))),'total':20,'configs':len(data),'time':time.time(),'pid':os.getpid()});print(n,flush=True)
 ranking=[{'key':k,**info[k],**m.summarize(v)} for k,v in data.items()];ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('per_image.json',data);dump('completion.json',{'time':time.time(),'configs':len(ranking),'baseline_percent':m.BASE,'control_masks_exact20':True,'source_channel_proof_verified':True,'R_per_fold_FP16_average_precision_limit':True,'best':ranking[0],'eligible_reduction_passes':sum(x['checkpoint_count']<15 and x['pass_mean_local'] for x in ranking),'release_replaced':False});print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
