import os,sys,json,hashlib,re,traceback,time,statistics
from pathlib import Path
from multiprocessing import get_context
import numpy as np
O=Path('/workspace/model349_compress/codex_allmodel_fold_audit_20261013');B=O/'baseline';PY=sys.executable
GLOBAL={}
def channel_order(c):
 f=c['folder'];v=c['variant'] or ''
 # Sources whose producers explicitly return the original main order.
 for prefix in ['probe','probe_p100','probe_high','probe_p100_high','probe_scale','probe_ctx','probe_tent','newmem','eut_spec','newmem_slots','slim']:
  if f.startswith('/workspace/model349_compress/'+prefix+'/'):return 'main','verified probe producer / original main-order source'
 if '/probabilities/' in f and f.split('/')[-1] in ['269','322','326','329','336','347']:return 'main','portable infer_lite output contract'
 if '/outputs/' in f:
  m=re.search(r'/outputs/model(\d+)',f)
  if m and int(m.group(1))<300:return 'main','original model1/advanced_inference class contract'
 # No guessing from the higher-scoring channel permutation.
 for folder in [Path(f),Path(f).parent,Path(f).parent.parent]:
  for name in ['metrics.json','evaluation.json','valid_metrics.json','summary.json']:
   p=folder/name
   if not p.is_file():continue
   try:
    d=json.loads(p.read_text());cs=d.get('class_names') if isinstance(d,dict) else None
    if cs and 'Al' in cs and len(cs)==4:return ('team' if cs[0]=='Al' else 'main'),'recorded class_names '+str(p)
   except Exception:pass
 return None,'class order requires source/metadata verification'
def cost(c):
 f=c['folder'];v=c['variant'] or ''
 if v.startswith('f') and re.match(r'f[1-4]_v\d+',v):return 1,'one explicit fold'
 if '/newmem/' in f or '/eut_spec/' in f or '/probe_scale/' in f or '/probe_ctx/' in f or '/probe_tent/' in f:return 1,'standalone member cache'
 if '/probe/' in f or '/probe_p100/' in f:
  name=f.split('/')[-1]
  if name in ['269','336','347','322soup','326soup','329soup']:return 1,'standalone/soup member'
 if '/outputs/' in f and not any(s in f.lower() for s in ['ensemble','oof','route','blend','joint','search']):
  folder=Path(f).parent if Path(f).suffix else Path(f)
  if any(folder.glob('*.pt')) or any(folder.glob('*.pth')):return 1,'standalone checkpoint adjacent to probability archive'
 return None,'member count must be resolved before compression claim'
def read(c,n):
 r=c['files'][n]
 if c['storage']=='npy':a=np.load(r['path'])
 else:
  with np.load(r['path'],allow_pickle=False) as z:a=z[r['key']]
 a=a.astype(np.float32)
 if c['channel_order']=='team':a=a[[1,2,0,3]]
 return a
def init():
 import torch,importlib.util
 from PIL import Image
 torch.set_num_threads(2);X=B/'l18h/parent/Model349_exh14';sys.path[:0]=[str(X),str(X/'src'),str(B/'Model349_lite20_b4ws_bundle/Model349_fusion805/refiner')]
 import infer_lite
 spec=importlib.util.spec_from_file_location('fu',B/'add/parent0/infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu)
 ref=fu.Residual().eval();ref.load_state_dict(torch.load(B/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.loads((B/'l18h/R_valid/input_manifest.json').read_text());pal=np.array(man['config']['palette'],np.uint8);records={}
 for item in man['images']:
  n=item['name'];ip=B/f'data/valid/images/{n}_image.png';assert hashlib.sha256(ip.read_bytes()).hexdigest()==item['sha256']
  rgb=np.array(Image.open(B/f'data/valid/masks/{n}_mask.png').convert('RGB'));gt=np.full(rgb.shape[:2],255,np.uint8)
  for c,col in enumerate(pal):gt[np.all(rgb==col,-1)]=c
  assert (gt<4).all();r=np.array(Image.open(B/f'l18h/e2e_valid/release/predicted_labels/{n}.png'));e=np.load(B/f'codex_lite16_b4_local_20261012/fresh/valid/expert/b4_13/{n}.npy').astype(np.float32);base=np.array(Image.open(B/f'codex_lite16_b4_local_20261012/fresh/valid/b4_13/fusion/predicted_labels/{n}.png'))
  hr={(s,f):np.load(B/f'add/pe/valid/hr392_s{s}_f{f}/{n}.npy').astype(np.float32)[[1,2,0,3]] for s,f in [(1,2),(1,4),(2,1),(2,4)]};h1=(hr[1,2]+hr[1,4])/2;h2=(hr[2,1]+hr[2,4])/2;b4=np.mean([np.load(B/f'add/pe/valid/model329_ws_f{f}/{n}.npy').astype(np.float32)[[1,2,0,3]] for f in [1,3]],0)
  sources={c:np.load(B/f'l18h/R_valid/probabilities/{c}/{n}.npy').astype(np.float32) for c in man['config']['inference']['active_components'] if c!='base'}
  import pickle
  with (B/f'l18h/R_valid/probabilities/base/{n}.pkl').open('rb') as f:it=pickle.load(f)
  assert np.array_equal(infer_lite.predict_fixed(it,sources,man['config']),r),'R baseline mismatch'
  gray=torch.from_numpy(fu.prepare_gray(np.array(Image.open(ip).convert('L'),dtype=np.float32)/255.)[None,None])
  with torch.no_grad():fv,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
  ex=fu.labels(e);fl=fu.labels(fv.numpy()[0]);pred=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8);assert np.array_equal(pred,base)
  records[n]={'gt':gt,'r':r,'e':e,'base':base,'hr':hr,'h1':h1,'h2':h2,'b4':b4,'sources':sources,'item':it,'gray':gray,'ex':ex,'fl':fl}
 GLOBAL.update(records=records,refiner=ref,fu=fu,infer=infer_lite,cfg=man['config'])
def iou(p,g):
 vals=[]
 for c in range(4):
  u=((p==c)|(g==c)).sum();vals.append(float(((p==c)&(g==c)).sum()/u) if u else 0.)
 return vals,float(np.mean(vals))
def task(c):
 import torch
 fu=GLOBAL['fu'];data={};digest=hashlib.sha256();folder=O/f'results/cache{c["id"]:04d}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 try:
  for n,rec in GLOBAL['records'].items():
   p=read(c,n);assert p.shape==rec['e'].shape,'non-native shape';assert np.isfinite(p).all() and np.min(p)>=-1e-6 and np.max(np.abs(p.sum(0)-1))<.04,'not normalized probabilities';digest.update(p.tobytes());e=rec['e'];r=rec['r'];base=rec['base'];g=rec['gt']
   configurations={'HR4_to_one':e+.5*p-.25*rec['h1']-.25*rec['h2'],'HRseed1_pair_to_one':e+.25*(p-rec['h1']),'HRseed2_pair_to_one':e+.25*(p-rec['h2']),'E_B4_pair_to_one':e+.5*(p-rec['b4'])};predictions={}
   for tag,mix in configurations.items():
    mix=np.maximum(mix,1e-7);mix/=mix.sum(0,keepdims=True)
    with torch.no_grad():v,_=GLOBAL['refiner'](torch.from_numpy(np.log(np.maximum(mix,1e-7))[None]),rec['gray'])
    ex=fu.labels(mix);fl=fu.labels(v.numpy()[0]);predictions[tag]=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8)
   for component,tag in [('329','R329_3_to_one'),('326','R326_2_to_one')]:
    sources=dict(rec['sources']);sources[component]=p;rr=GLOBAL['infer'].predict_fixed(rec['item'],sources,GLOBAL['cfg']);predictions[tag]=np.where(rec['fl']==2,2,np.where(np.isin(rec['ex'],[0,2]),rec['ex'],rr)).astype(np.uint8)
   bci,bmi=iou(base,g);singleci,single=iou(fu.labels(p),g);data[n]={'single_miou':single,'single_class_iou':singleci,'candidates':{}}
   for tag,pred in predictions.items():
    ci,mi=iou(pred,g);data[n]['candidates'][tag]={'miou':mi,'delta_pp':100*(mi-bmi),'class_delta_pp':(100*(np.array(ci)-bci)).tolist()}
  summary={}
  for tag in next(iter(data.values()))['candidates']:
   ds=[r['candidates'][tag]['delta_pp'] for r in data.values()];ci=np.array([r['candidates'][tag]['class_delta_pp'] for r in data.values()]);removed={'HR4_to_one':4,'HRseed1_pair_to_one':2,'HRseed2_pair_to_one':2,'E_B4_pair_to_one':2,'R329_3_to_one':3,'R326_2_to_one':2}[tag]
   summary[tag]={'percent':80.55464328764197+float(np.mean(ds)),'delta_pp':float(np.mean(ds)),'worst_image_delta_pp':min(ds),'worst_image_class_delta_pp':float(ci.min()),'mean_class_delta_pp':ci.mean(0).tolist(),'better':sum(v>1e-12 for v in ds),'worse':sum(v< -1e-12 for v in ds),'checkpoint_count_if_no_reuse':16-removed+c['weight_cost'] if c['weight_cost'] is not None else None,'pass_mean_local':bool(np.mean(ds)>=-1e-12 and min(ds)>=-.1)}
  result={'cache':c,'probability_sha256':digest.hexdigest(),'single_percent':float(np.mean([r['single_miou'] for r in data.values()])*100),'summary':summary,'per_image':data,'valid_development_screening':True,'full_E_family_fresh':False};(folder/'result.json').write_text(json.dumps(result,indent=1));return result
 except Exception:return {'cache':c,'error':traceback.format_exc()}
def main():
 # Wait for the verified baseline transfer; don't race partially written tar outputs.
 while not (O/'baseline_ready.json').exists():time.sleep(10)
 inv=json.loads((O/'inventory.json').read_text());cs=[];coverage=[]
 for i,c in enumerate(inv['probability_caches']):
  c=dict(c,id=i);order,proof=channel_order(c);num,costproof=cost(c);c.update(channel_order=order,channel_proof=proof,weight_cost=num,weight_cost_proof=costproof);coverage.append(c)
  if c['complete20'] and order is not None:cs.append(c)
 (O/'coverage.json').write_text(json.dumps(coverage,indent=1));(O/'queue_status.json').write_text(json.dumps({'state':'CPU exhaustive fold-group substitutions','candidate_caches':len(cs),'targets_per_cache':6,'pid':os.getpid()}));results=[]
 with get_context('spawn').Pool(2,initializer=init) as pool:
  for result in pool.imap_unordered(task,cs):
   results.append(result);(O/'progress.json').write_text(json.dumps({'completed':len(results),'total':len(cs),'last_cache':result['cache']['id'],'failures':sum('error' in r for r in results)}));print(len(results),'/',len(cs),'cache',result['cache']['id'],'error' if 'error' in result else 'ok',flush=True)
 (O/'all_results.json').write_text(json.dumps(results,indent=1));unique={};errors=[]
 for r in results:
  if 'error' in r:errors.append(r);continue
  unique.setdefault(r['probability_sha256'],[]).append(r)
 ranking=[]
 for group in unique.values():
  r=group[0]
  for tag,v in r['summary'].items():ranking.append({'cache_id':r['cache']['id'],'source':r['cache']['folder'],'variant':r['cache']['variant'],'target':tag,'exact_probability_duplicate_cache_ids':[q['cache']['id'] for q in group],**v})
 ranking.sort(key=lambda r:r['delta_pp'],reverse=True);(O/'ranking.json').write_text(json.dumps(ranking,indent=2));(O/'completion.json').write_text(json.dumps({'candidate_caches_measured':len(results),'unique_probability_sets':len(unique),'errors':len(errors),'unverified_channel_caches':sum(c['channel_order'] is None for c in coverage),'checkpoints_without_eligible_cache_not_inferred':True,'release_replaced':False,'time':time.time()}));print('COMPLETE',len(unique),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:(O/'failure.json').write_text(json.dumps({'trace':traceback.format_exc()}));raise
