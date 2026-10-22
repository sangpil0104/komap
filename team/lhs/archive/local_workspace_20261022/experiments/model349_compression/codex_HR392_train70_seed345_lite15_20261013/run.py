from pathlib import Path
import os,sys,json,time,subprocess,traceback,gc
import numpy as np
K=Path.home()/'komap';O=K/'codex_HR392_train70_seed345_lite15_20261013';OLD=K/'codex_HR392_train70_N3_20261013'
sys.path.insert(0,str(K/'codex_lite15_HRtrain70_evaluation_20261013'))
import pipeline_common as pc
SEEDS=[3,4,5];ROLES={'single_s2f1':('s2f1',),'pair_s2f1_s2f4':('s2f1','s2f4')}
def dump(name,value):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):dump('status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
  stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:status('waiting GPU0 idle');time.sleep(10)
def train(seed):
 import torch
 out=O/f'training/seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():
  done=json.load(open(out/'completion.json'));assert pc.sha(out/'last.pt')==done['checkpoint_sha'];return out
 idle();status('train70 HR392',seed=seed,epochs=70)
 # Copy the exact original argv, changing only output path and seed.
 recipe=json.load(open(OLD/'training/seed1/recipe.json'));cmd=list(recipe['command']);cmd[0]=sys.executable
 cmd[cmd.index('--output-dir')+1]=str(out);cmd[cmd.index('--seed')+1]=str(seed)
 assert cmd[cmd.index('--val-every')+1]=='0' and cmd[cmd.index('--epochs')+1]=='70'
 if (out/'train_state.pt').exists():cmd.append('--resume')
 elif (out/'last.pt').exists():raise RuntimeError('unverified partial checkpoint; inspect before retry')
 dump(f'training/seed{seed}/recipe.json',dict(command=cmd,train70=True,no_valid_epoch_selection=True,original_recipe=recipe,changed_fields=['output-dir','seed']))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
 with (out/'train.log').open('a') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  dump('active_child.json',dict(pid=child.pid,seed=seed,stage='training',log=str(out/'train.log'),time=time.time()))
  code=child.wait()
 assert code==0,f'training seed{seed} exited {code}'
 state=torch.load(out/'train_state.pt',map_location='cpu',weights_only=False);assert state['epoch']==70
 dump(f'training/seed{seed}/completion.json',dict(epoch=70,checkpoint_sha=pc.sha(out/'last.pt')));del state;gc.collect();return out

def infer(seed,out):
 import torch
 sys.path.insert(0,str(pc.P/'model397_reproduction_bundle/model392'))
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 dest=O/f'evaluation/seed{seed}';prob=dest/'probabilities';prob.mkdir(parents=True,exist_ok=True)
 h=pc.sha(out/'last.pt')
 if (dest/'completion.json').exists():
  done=json.load(open(dest/'completion.json'));assert done['checkpoint_sha']==h;return
 idle();status('valid20 flip4 inference',seed=seed)
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,saved=load_model_and_preprocessing(out/'last.pt',torch.device('cuda'));prep=resolve_preprocessing(saved);assert prep['normalize_width']
 ds=KomapFullDataset(K/'data/valid/images',masks_dir=None);assert len(ds)==20
 inputs={}
 for i,pair in enumerate(ds.pairs):
  n=pair['stem'];f=prob/f'{n}.npy';ip=K/f'data/valid/images/{n}_image.png';inputs[str(ip)]=pc.sha(ip)
  if f.exists():
   v=np.load(f);assert np.isfinite(v).all();continue
  sample=ds[i];scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
  v=predict_probs_tta(model,sample['image'],patch_size=512,overlap=scaled_overlap(512,2.,170,scale),batch_size=2,tta=True,input_scale=scale,device=torch.device('cuda'),tta_mode='flip4').numpy().astype(np.float32)
  assert np.isfinite(v).all();tmp=f.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(f);print('valid',seed,n,flush=True)
 for p,h0 in inputs.items():assert pc.sha(p)==h0
 del model;torch.cuda.empty_cache();gc.collect()
 dump(f'evaluation/seed{seed}/completion.json',dict(images=20,checkpoint_sha=h,preprocessing=prep,input_sha=inputs,channel_order='team',tta='flip4'))

def evaluate(seeds):
 from PIL import Image
 G=pc.init_records();data={};checks={};comparisons={}
 for seed in seeds:
  root=OLD if seed in [1,2] else O;ck=root/f'training/seed{seed}/last.pt';checks[str(ck)]=pc.sha(ck)
  if seed in [1,2]:assert checks[str(ck)]==json.load(open(root/f'training/seed{seed}/completion.json'))['checkpoint_sha']
  for n,r in G['records'].items():
   p=np.load(root/f'evaluation/seed{seed}/probabilities/{n}.npy').astype(np.float32)[[1,2,0,3]]
   assert p.shape==r['e'].shape and np.isfinite(p).all()
   for role,targets in ROLES.items():
    parts=dict(r['parts']);parts.update({t:p for t in targets});pred=pc.final(G,r,r['r'],pc.expert(parts,r['b4']));key=f'seed{seed}_{role}'
    data.setdefault(key,{})[n]=pc.score(r,pred);dest=O/f'predictions/{key}';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(dest/f'{n}.png')
  if seed in [1,2]:
   prior=json.load(open(K/'codex_lite15_HRtrain70_evaluation_20261013/per_image.json'))
   for role,targets in ROLES.items():
    key=f'seed{seed}_{role}';oldkey=f'seed{seed}_'+('_'.join(targets))
    for n,row in data[key].items():
     delta=row['percent']-prior[oldkey][n]['percent'];comparisons.setdefault(key,{})[n]=delta
     # Keep prior-score drift visible; all five seeds use this same baseline/code.
 ranking=[dict(key=key,checkpoint_count=14 if '_pair_' in key else 15,**pc.summary(rows)) for key,rows in data.items()]
 summary={}
 for role in ROLES:
  rows=[x for x in ranking if x['key'].endswith(role)];vals=[x['percent'] for x in rows]
  spreads={n:dict(mean_delta_pp=float(np.mean([data[f'seed{s}_{role}'][n]['delta_pp'] for s in seeds])),sd_pp=float(np.std([data[f'seed{s}_{role}'][n]['delta_pp'] for s in seeds],ddof=1)),range_pp=float(np.ptp([data[f'seed{s}_{role}'][n]['delta_pp'] for s in seeds]))) for n in G['records']}
  summary[role]=dict(seeds=list(seeds),mean_percent=float(np.mean(vals)),sample_sd_pp=float(np.std(vals,ddof=1)),min_percent=min(vals),max_percent=max(vals),range_pp=float(np.ptp(vals)),gate_pass_count=sum(x['pass_mean_local'] for x in rows),per_image_seed_spread=spreads)
 dump('prior_seed12_score_comparison.json',dict(per_image_difference_pp=comparisons,max_abs_difference_pp=max(abs(v) for rows in comparisons.values() for v in rows.values()),prior_scores_exact=False,all_seeds_rescored_with_identical_current_code=True));dump('ranking.json',ranking);dump('per_image.json',data);dump('seed_variability.json',summary);dump('scoring_receipt.json',dict(baseline_percent=pc.BASE,baseline_exact20=True,checkpoint_sha=checks,roles=ROLES,coefficients_fixed=True,release_replaced=False,test_GT_used=False))
 del G;gc.collect()

def main():
 O.mkdir(exist_ok=True)
 if (O/'completion.json').exists():return
 import torch
 torch.set_num_threads(2)
 inputs=json.load(open(OLD/'inputs_sha.json'))
 for p,h in inputs.items():assert pc.sha(p)==h,'original input changed '+p
 dump('source_receipt.json',dict(original_inputs_sha=inputs,original_seed12_preserved=True,primary_role='pair_s2f1_s2f4',new_seeds=SEEDS))
 # Verify original seed1/2 on lite15 before new GPU training.
 if not (O/'preflight.json').exists():
  status('baseline and original seed1/2 preflight');evaluate([1,2]);dump('preflight.json',dict(baseline_exact20=True,prior_seed12_replay_differences_recorded=True,all_seeds_common_rescoring=True))
 for seed in SEEDS:
  out=train(seed);infer(seed,out);evaluate(list(range(1,seed+1)));status('seed completed',seed=seed)
 for p,h in inputs.items():assert pc.sha(p)==h
 dump('completion.json',dict(seeds=[1,2,3,4,5],new_training_seeds=SEEDS,valid_epoch_selection=False,input_sha_unchanged=True,release_replaced=False,test_GT_used=False,time=time.time()));status('complete')
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',dict(time=time.time(),trace=traceback.format_exc()));raise
