"""Single bounded pilot, queued after preceding timing, fixed distill9 selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import fcntl,hashlib,importlib.util,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path('/workspace/model349_compress/base12_partial_distill329_20261008');PREVIOUS=ROOT.parent/'base12_singleworker_20261008'
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module('resume',PREVIOUS/'resume_single.py');u=s.u;u.ROOT=ROOT;u.idle=s.stable_idle;torch.set_num_threads(2)
def execute_py(script,args,log):
 u.idle();uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PILOT_GPU_UUID=uuid)
 with log.open('x') as stream:subprocess.run([u.PYTHON,'-B','-u',str(ROOT/script),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
def make_bundle(key,checkpoint):
 bundle=u.build_bundle(key);cfgpath=bundle/'configs/model349_lite.yaml';cfg=yaml.safe_load(cfgpath.read_text());cfg['inference']['b4_fold_ids']['329']=[1];cfg['experimental_student329']={'single_student':True,'not_original_fold1_selection':True,'checkpoint_sha256':u.sha(checkpoint),'protocol':'partial-distill329 fixed400 pilot'};cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
 manifest=u.read(bundle/'checkpoint_manifest.json');rows=[]
 for row in manifest['checkpoints']:
  p=bundle/row['path']
  if 'model329_fullstats_width4000_uniform_group_oof' in row['path']:
   p.unlink()
   if '/fold1/' not in row['path']:continue
   shutil.copy2(checkpoint,p);row=dict(row,sha256=u.sha(p),bytes=p.stat().st_size,source='synthetic/student pilot, not original fold1')
  rows.append(row)
 manifest.update(checkpoints=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows));assert manifest['count']==9;u.write(bundle/'checkpoint_manifest.json',manifest)
 files=[p for p in bundle.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.json'] and p.name!='reproduction_hashes.json' and '__pycache__' not in p.parts]
 u.write(bundle/'reproduction_hashes.json',{'files':{str(p.relative_to(bundle)):u.sha(p) for p in sorted(files)}})
 return bundle

def main():
 while not any((PREVIOUS/n).exists() for n in ['completion.json','failure.json']):time.sleep(5)
 if (PREVIOUS/'failure.json').exists():u.write(ROOT/'skipped.json',{'reason':'preceding worker experiment failed; do not overlap or launch training'});return
 assert u.read(PREVIOUS/'completion.json')['base_and_weights_unchanged']
 locks=[]
 for n in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008',PREVIOUS.name,ROOT.name]:
  f=(ROOT.parent/n/'gpu7.lock').open('a')
  for attempt in range(16):
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if attempt==15:raise
    time.sleep(2)
  locks.append(f)
 if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py|[t]rain_pilot.py'],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing own inference/training')
 u.idle();frozen=u.snapshot();u.write(ROOT/'frozen_before.json',frozen);assert frozen==u.read(PREVIOUS/'frozen_after.json')
 for name in ['training','cached_diagnostics','runs']:(ROOT/name).mkdir(exist_ok=False)
 monitor=(ROOT.parent/'base12_followup_20261008/monitor.py').read_text().replace(str(ROOT.parent/'base12_followup_20261008'),str(ROOT));(ROOT/'monitor.py').write_text(monitor)
 subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 train=module('pilottrain',ROOT/'train_pilot.py');p=ROOT/'training/init';p.mkdir();state=train.mean_state();state={k:v.half() if v.is_floating_point() else v for k,v in state.items()};torch.save(state,p/'last.pt');u.write(p/'provenance.json',{'uniform_mean':.25,'integer_buffers':'max','teacher_sha256':{str(d/'last.pt'):u.sha(d/'last.pt') for d in train.paths()},'training_started':False})
 # Reproduce the entire existing base's cached PNG/GT scoring before changing329.
 infer=module('baseinfer',u.BASE/'infer_lite.py');baseline=ROOT/'cached_diagnostics/base12_replay';baseline.mkdir();shutil.copy2(u.OLD/'input_manifest.json',baseline/'input_manifest.json');(baseline/'probabilities').symlink_to(u.OLD/'probabilities',target_is_directory=True);infer.assemble(baseline,u.GT)
 reference=u.metrics(u.OLD);replayed=u.metrics(baseline);assert reference==replayed,'Frozen cached baseline metrics differ'
 from PIL import Image
 for a,b in zip(sorted((u.OLD/'predicted_labels').glob('*.png')),sorted((baseline/'predicted_labels').glob('*.png'))):assert a.name==b.name and np.array_equal(np.array(Image.open(a)),np.array(Image.open(b)))
 u.write(ROOT/'baseline_replay_verification.json',{'all20_masks_and_all_class_iou_exact':True})
 bundles={'init9':make_bundle('init9',p/'last.pt')}
 execute_py('eval_branch.py',['--candidate','init9'],ROOT/'init9_branch.log')
 for mode in ['hard','distill']:
  execute_py('train_pilot.py',['--mode',mode],ROOT/(mode+'_train.log'))
  assert u.read(ROOT/'training'/mode/'summary.json')['completed']
  key=mode+'9';bundles[key]=make_bundle(key,ROOT/'training'/mode/'last.pt');execute_py('eval_branch.py',['--candidate',key],ROOT/(key+'_branch.log'))
 assert u.read(ROOT/'training/hard/crop_sequence.json')==u.read(ROOT/'training/distill/crop_sequence.json'),'Training crops differ'
 diagnostics={}
 for key in bundles:
  m=u.metrics(ROOT/'cached_diagnostics'/key);diagnostics[key]={'metrics':m,'vs_base12':u.pair(reference,m),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),m),'checkpoints':u.read(bundles[key]/'checkpoint_manifest.json')}
 u.write(ROOT/'cached_comparison.json',diagnostics);raw={};timings=[];tests=[]
 selected=diagnostics['distill9'];passes=selected['vs_base12']['quality_pass'] and selected['vs_lite15']['quality_pass']
 if passes:
  run=u.execute(bundles['distill9'],'distill9_valid20');m=u.metrics(run);raw['distill9']={'metrics':m,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(reference,m),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),m),'checkpoints':selected['checkpoints']};u.write(ROOT/'raw_comparison.json',raw);passes=raw['distill9']['vs_base12']['quality_pass'] and raw['distill9']['vs_lite15']['quality_pass']
  if passes:
   control=u.build_bundle('control12')
   for index,order in enumerate([['control12','distill9'],['distill9','control12'],['control12','distill9']],1):
    for key in order:
     run=u.execute(control if key=='control12' else bundles[key],f'timing_{index}_{key}');m=u.metrics(run);timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base12':u.pair(reference,m),'vs_lite15':u.pair(u.metrics(u.OLD.parent/'cpu2_lite15'),m)});u.write(ROOT/'timing_runs.json',timings)
   u.execute(bundles['distill9'],'distill9_test10',test=True);tests=['distill9']
 after=u.snapshot();assert after==frozen;u.write(ROOT/'frozen_after.json',after)
 for mode in ['hard','distill']:
  for row in u.read(ROOT/'training'/mode/'training_manifest.json')['rows']:assert u.sha(Path(row['image_path']))==row['image_sha256'] and u.sha(Path(row['mask_path']))==row['mask_sha256']
 u.write(ROOT/'completion.json',{'pilot_completed':True,'cached_diagnostics':['init9','hard9','distill9'],'full_raw_valid_candidates':list(raw),'time_candidate':'distill9' if timings else None,'test_candidates':tests,'base_and_weights_unchanged':True,'training_images_unchanged':True,'training_crop_sequence_exact':True,'configured_updates_per_trial':400,'official_valid_used_in_training':False,'independent_generalization_verified':False,'selection_rule':'only fixed distill9; controls not chosen'})
 print('SESSION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
