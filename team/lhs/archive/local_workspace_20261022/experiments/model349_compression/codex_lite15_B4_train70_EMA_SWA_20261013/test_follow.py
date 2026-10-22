from pathlib import Path
import os,sys,json,time,traceback,subprocess,copy,importlib.util
import numpy as np
import pipeline_common as pc
import test_common as tc
K=pc.K;P=pc.P;X=pc.X
KIND=sys.argv[1];DIRS={'HR':K/'codex_lite15_HRtrain70_evaluation_20261013','B4':K/'codex_lite15_B4_train70_EMA_SWA_20261013','sharing':K/'codex_lite15_model_sharing_test_20261013'};O=DIRS[KIND]/'test_followup';O.mkdir(parents=True,exist_ok=True)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def main():
 if (O/'completion.json').exists():return
 dependency=DIRS[KIND]/('sharing_ready.json' if KIND=='sharing' else 'completion.json')
 while not dependency.exists() or not (pc.A/'fresh_completion.json').exists() or (KIND=='sharing' and not (DIRS['B4']/'test_followup/completion.json').exists()):
  if (DIRS[KIND]/'failure.json').exists() or (pc.A/'fresh_failure.json').exists():raise RuntimeError('prior validation failed')
  dump('status.json',{'stage':'waiting preceding validation/fresh inference','time':time.time(),'kind':KIND});time.sleep(20)
 ranking=json.load(open(DIRS[KIND]/'ranking.json'));passing=[r for r in ranking if r['pass_mean_local']];selected=[]
 if KIND=='B4':
  for role in ['R329_to_one','E_B4pair_to_one','R329_and_E_B4_shared']:
   options=[r for r in passing if r['role']==role]
   if options:selected.append(options[0])
 for row in passing:
  if row not in selected:selected.append(row)
  if len(selected)>=6:break
 if not selected:dump('completion.json',{'time':time.time(),'test_inference_skipped':True,'reason':'no candidate met predeclared mean/local validation conditions','tested_valid_candidates':len(ranking),'GT_test_used':False});return
 import torch
 from PIL import Image
 from scipy import ndimage as ndi
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;G=tc.records(O);sys.path.insert(0,str(X/'src'));from b4_zcdf import load_member,load_gray,predict_probs
 sp=importlib.util.spec_from_file_location('warm',K/'codex_lite15_B4_train70_EMA_SWA_20261013/run.py');warm=importlib.util.module_from_spec(sp);sp.loader.exec_module(warm)
 probabilities={};checked={}
 def b4_prob(tag,folder,file,dual=False):
  dest=O/f'probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True);h=pc.sha(folder/file);checked[str(folder/file)]=h
  if not (dest/'completion.json').exists():
   idle();member=load_member(folder,'cuda',file)
   for n in G['records']:
    if (dest/f'{n}_d4.npy').exists() and (not dual or (dest/f'{n}_single.npy').exists()):continue
    gray=load_gray(K/f'data/test/images/{n}_image.png')
    if dual:p0,p8=warm.predict_dual(member,gray);np.save(dest/f'{n}_single.npy',p0)
    else:p8=predict_probs([member],gray,tta=True,device='cuda').cpu().numpy().astype(np.float32)[[1,2,0,3]]
    np.save(dest/f'{n}_d4.npy',p8)
   del member;torch.cuda.empty_cache();(dest/'completion.json').write_text(json.dumps({'sha256':h,'images':10}))
  return dest
 for row in selected:
  if KIND=='B4':
   p=Path(row['checkpoint']);assert pc.sha(p)==row['checkpoint_sha256'];probabilities[row['candidate']]=b4_prob(row['candidate'],p.parent,p.name,True)
  elif KIND=='HR':
   seed=int(row['key'].split('_')[0].removeprefix('seed'));tag=f'HRseed{seed}';dest=O/f'probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True);p=K/f'codex_HR392_train70_N3_20261013/training/seed{seed}/last.pt';checked[str(p)]=pc.sha(p)
   if not (dest/'completion.json').exists():
    idle();sys.path.insert(0,str(P/'model397_reproduction_bundle/model392'));from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing;from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap;from komap_seg.inference import predict_probs_tta
    model,saved=load_model_and_preprocessing(p,torch.device('cuda'));prep=resolve_preprocessing(saved);ds=KomapFullDataset(K/'data/test/images',masks_dir=None)
    for i,pair in enumerate(ds.pairs):
     n=pair['stem'];f=dest/f'{n}.npy'
     if f.exists():continue
     sample=ds[i];scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width']);v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=scaled_overlap(512,2.,170,scale),batch_size=2,tta=True,input_scale=scale,device=torch.device('cuda'),tta_mode='flip4').numpy().astype(np.float32)[[1,2,0,3]];np.save(f,v)
    del model;torch.cuda.empty_cache();(dest/'completion.json').write_text(json.dumps({'images':10,'checkpoint_sha256':checked[str(p)]}))
   probabilities[tag]=dest
  elif row.get('target') in [1,3]:
   other=3 if row['target']==1 else 1;probabilities[f'EB4fold{other}']=b4_prob(f'EB4fold{other}',pc.A/f'assets/model329_ws/fold{other}','last.pt')
 rows={}
 for n,r in G['records'].items():
  donors={k:r['sources'][k] for k in ['326','322','329','347','269']};donors['N3']=r['parts']['N3']
  for case in selected:
   key=case['key'];rr=None;e=r['e'];src=dict(r['sources']);cfg=G['cfg']
   if KIND=='B4':
    dest=probabilities[case['candidate']];p0=pc.rt(np.load(dest/f'{n}_single.npy'));p8=pc.rt(np.load(dest/f'{n}_d4.npy'))
    if case['role'] in ['R329_to_one','R329_and_E_B4_shared']:src['329']=p0
    if case['role'] in ['E_B4pair_to_one','R329_and_E_B4_shared']:e=tc.expert_delta(r,b4=p8)
   elif KIND=='HR':
    seed=int(key.split('_')[0].removeprefix('seed'));p=pc.rt(np.load(probabilities[f'HRseed{seed}']/f'{n}.npy').astype(np.float32));parts=dict(r['parts']);parts.update({t:p for t in key.split('_')[1:]});e=tc.expert_delta(r,parts=parts)
   else:
    if case.get('R329_group_shared_from'):src['329']=pc.rt(donors[case['R329_group_shared_from']])
    if case.get('R329_removed_without_replacement'):cfg=copy.deepcopy(cfg);cfg['route']['gamma329']=[0,0,0,0]
    if case.get('E_B4_pair_shared_from'):e=tc.expert_delta(r,b4=pc.rt(donors[case['E_B4_pair_shared_from']]))
    if case.get('E_B4_slot_shared_from'):
     donor=donors[case['E_B4_slot_shared_from']]
     if case['target']=='pair':b=pc.rt(donor)
     else:
      other=3 if case['target']==1 else 1;b=pc.rt((donor+np.load(probabilities[f'EB4fold{other}']/f'{n}_d4.npy'))/2)
     e=tc.expert_delta(r,b4=b)
    if case.get('E_HR_slots_shared_from'):
     parts=dict(r['parts']);parts.update({t:donors[case['E_HR_slots_shared_from']] for t in case['targets']});e=tc.expert_delta(r,parts=parts)
   rr=G['infer'].predict_fixed(r['item'],src,cfg);pred=pc.final(G,r,rr,e);diff=pred!=r['base'];labels,_=ndi.label(diff,structure=np.ones((3,3)));sizes=np.bincount(labels.ravel())[1:];trans=np.bincount((r['base']*4+pred)[diff],minlength=16).reshape(4,4)
   rows.setdefault(key,[]).append({'image':n,'changed_pixels':int(diff.sum()),'total_pixels':diff.size,'changed_percent':100*float(diff.mean()),'transitions':trans.tolist(),'largest_changed_region_px':int(sizes.max()) if len(sizes) else 0,'class_area_delta_px':[int((pred==c).sum()-(r['base']==c).sum()) for c in range(4)]});dest=O/f'results/{key}/predicted_labels';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(dest/f'{n}.png')
 for p,h in {**G['sha'],**checked}.items():assert pc.sha(p)==h
 summary={k:{'changed_pixels':sum(x['changed_pixels'] for x in v),'total_pixels':sum(x['total_pixels'] for x in v),'changed_percent':100*sum(x['changed_pixels'] for x in v)/sum(x['total_pixels'] for x in v)} for k,v in rows.items()};dump('comparison.json',{'kind':KIND,'selected_validation_candidates':selected,'summary':summary,'per_image':rows,'current15_test_control_exact10':True,'test_GT_used':False,'test_accuracy_claimed':False,'input_SHA_unchanged':True});dump('completion.json',{'time':time.time(),'cases':len(rows),'images':10,'GT_test_used':False,'release_replaced':False})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
