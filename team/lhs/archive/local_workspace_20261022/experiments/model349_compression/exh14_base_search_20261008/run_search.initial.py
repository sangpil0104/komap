"""Fixed storage and execution search against user-selected exh14, single GPU7."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import ast,copy,fcntl,importlib.util,shutil,subprocess,time
from pathlib import Path
import torch,yaml
ROOT=Path('/workspace/model349_compress/exh14_base_search_20261008');SOURCE=ROOT.parent/'Model349_exh14'
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module('single_resume',ROOT.parent/'base12_singleworker_20261008/resume_single.py');e=s.s.e;u=s.u;u.ROOT=ROOT;u.idle=s.stable_idle;torch.set_num_threads(2)
MANIFEST=u.read(ROOT/'base14_checkpoint_manifest.json')
def tensors(obj,key='root'):
 if isinstance(obj,torch.Tensor):return {key:obj}
 if isinstance(obj,dict):return {p:t for k,v in obj.items() for p,t in tensors(v,key+'/'+str(k)).items()}
 if isinstance(obj,(list,tuple)):return {p:t for k,v in enumerate(obj) for p,t in tensors(v,key+'/'+str(k)).items()}
 return {}
def convert(obj):
 if isinstance(obj,torch.Tensor):return obj.half() if obj.is_floating_point() else obj.clone()
 if isinstance(obj,dict):
  out=obj.__class__((k,convert(v)) for k,v in obj.items())
  if hasattr(obj,'_metadata'):out._metadata=copy.deepcopy(obj._metadata)
  return out
 if isinstance(obj,list):return [convert(v) for v in obj]
 if isinstance(obj,tuple):return tuple(convert(v) for v in obj)
 return obj
def freeze():
 files=[SOURCE/r['path'] for r in MANIFEST['checkpoints']]+[SOURCE/'infer_lite.py']+list((SOURCE/'configs').glob('*'))+list((SOURCE/'src').rglob('*.py'))
 return {str(p.relative_to(SOURCE)):u.sha(p) for p in files if p.is_file()}
def hashes(bundle):
 files=[p for p in bundle.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml'] and p.name!='reproduction_hashes.json' and '__pycache__' not in p.parts]
 u.write(bundle/'reproduction_hashes.json',{'files':{str(p.relative_to(bundle)):u.sha(p) for p in sorted(files)}})
def build(key,parent=None):
 src=parent or SOURCE;bundle=ROOT/'bundles'/key;bundle.mkdir(parents=True,exist_ok=False)
 for name in ['src','configs']:shutil.copytree(src/name,bundle/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(src/'infer_lite.py',bundle/'infer_lite.py')
 manifest=u.read(src/'checkpoint_manifest.json') if parent else copy.deepcopy(MANIFEST)
 for row in manifest['checkpoints']:
  p=bundle/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(src/row['path'])
  q=(src/row['path']).parent/'args.json'
  if q.exists():shutil.copy2(q,p.parent/'args.json')
 u.write(bundle/'checkpoint_manifest.json',manifest);hashes(bundle);return bundle
def fp16(bundle):
 rows=[]
 for row in u.read(bundle/'checkpoint_manifest.json')['checkpoints']:
  p=bundle/row['path'];source=p.resolve();obj=torch.load(source,map_location='cpu',weights_only=False);target=convert(obj);original=tensors(obj);converted=tensors(target);assert original.keys()==converted.keys() and original
  errors={k:float((v.float()-converted[k].float()).abs().max()) for k,v in original.items() if v.is_floating_point() and v.numel()}
  temp=p.parent/'fp16.tmp';torch.save(target,temp);loaded=tensors(torch.load(temp,map_location='cpu',weights_only=False));assert loaded.keys()==original.keys()
  assert all(torch.equal(loaded[k],v.half() if v.is_floating_point() else v) for k,v in original.items())
  rows.append({'path':row['path'],'source_sha256':u.sha(source),'source_bytes':source.stat().st_size,'all_stored_tensors_equal_expected_cast':True,'floating_tensor_count':len(errors),'max_absolute_rounding_error':max(errors.values(),default=0),'tensor_count':len(original)})
  p.unlink();temp.rename(p)
 u.write(bundle/'fp16_conversion_verification.json',rows);e.update_manifest(bundle);e.deflate(bundle)
 p=bundle/'configs/model349_lite.yaml';cfg=yaml.safe_load(p.read_text());cfg['inference']['weight_storage_dtype']='float16; load into float32 parameters';p.write_text(yaml.safe_dump(cfg,sort_keys=False));hashes(bundle)
def single_worker(bundle):
 tree=ast.parse((ROOT.parent/'base12_singleworker_20261008/run_single.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');after=None
 for i,n in enumerate(main.body):
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and n.value.value=='def worker(work, component):\n    import torch':after=main.body[i+1].value.value
 assert after is not None
 p=bundle/'infer_lite.py';code=p.read_text();before='def worker(work, component):\n    import torch';assert code.count(before)==1;code=code.replace(before,after)
 assert code.count('    pending = list(active_components(config))')==1;code=code.replace('    pending = list(active_components(config))',"    if len(gpus) != 1: raise ValueError('one GPU required')\n    pending = ['all']")
 code=code.replace('choices=COMPONENTS, help=argparse.SUPPRESS',"choices=COMPONENTS + ['all'], help=argparse.SUPPRESS")
 code=code.replace('with tqdm(total=len(active_components(config)),','with tqdm(total=len(pending),')
 compile(code,str(p),'exec');p.write_text(code);hashes(bundle)
def main():
 locks=[]
 for name in ['audit_20261008','base12_followup_20261008','base12_inputcache_20261008','base12_highflip4_20261008','base12_resident_packed_20261008','base12_singleworker_20261008','base12_partial_distill329_20261008',ROOT.name]:
  f=(ROOT.parent/name/'gpu7.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py|[t]rain_pilot.py'],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing inference; no duplicate launch')
 u.idle();frozen=freeze();assert all(frozen[r['path']]==r['sha256'] and (SOURCE/r['path']).stat().st_size==r['bytes'] for r in MANIFEST['checkpoints']);assert frozen['infer_lite.py']==u.read(ROOT/'existing_raw_completion.json')['inference_sha256'];u.write(ROOT/'frozen_before.json',frozen)
 (ROOT/'runs').mkdir(exist_ok=False);control=build('control14');u.BASE=control
 monitor=(ROOT.parent/'base12_followup_20261008/monitor.py').read_text().replace(str(ROOT.parent/'base12_followup_20261008'),str(ROOT));(ROOT/'monitor.py').write_text(monitor);subprocess.Popen([u.PYTHON,'-B',str(ROOT/'monitor.py')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 base=u.execute(control,'control14_valid20');u.OLD=base;reference=u.metrics(base)
 bundles={'fp16_storage14':build('fp16_storage14')};fp16(bundles['fp16_storage14']);results={}
 def evaluate(key):
  run=u.execute(bundles[key],key+'_valid20');m=u.metrics(run);results[key]={'metrics':m,'completion':u.read(run/'completion.json'),'vs_base14':u.pair(reference,m),'vs_base12':u.pair(u.metrics(ROOT.parent/'audit_20261008/runs/cpu2_fp16_12'),m),'checkpoints':u.read(bundles[key]/'checkpoint_manifest.json')};u.write(ROOT/'raw_comparison.json',results)
 evaluate('fp16_storage14')
 if results['fp16_storage14']['vs_base14']['quality_pass']:
  bundles['fast14']=build('fast14',bundles['fp16_storage14']);e.resident(bundles['fast14']);e.hramp(bundles['fast14']);single_worker(bundles['fast14']);hashes(bundles['fast14'])
  probe=(ROOT.parent/'base12_resident_packed_20261008/probe.py').read_text().replace(str(ROOT.parent/'base12_resident_packed_20261008'),str(ROOT)).replace('bundles/resident12','bundles/fast14');(ROOT/'probe.py').write_text(probe)
  u.idle()
  with (ROOT/'probe.log').open('x') as log:subprocess.run([u.PYTHON,'-B',str(ROOT/'probe.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='7'),stdout=log,stderr=subprocess.STDOUT,check=True)
  evaluate('fast14')
 selected=next((key for key in ['fast14','fp16_storage14'] if key in results and results[key]['vs_base14']['quality_pass']),None);timings=[]
 if selected:
  for index,order in enumerate([['control14',selected],[selected,'control14'],['control14',selected]],1):
   for key in order:
    run=u.execute(control if key=='control14' else bundles[key],f'timing_{index}_{key}');m=u.metrics(run);timings.append({'run':run.name,'candidate':key,'completion':u.read(run/'completion.json'),'vs_base14':u.pair(reference,m)});u.write(ROOT/'timing_runs.json',timings)
  u.execute(bundles[selected],selected+'_test10',test=True)
 after=freeze();assert after==frozen;u.write(ROOT/'frozen_after.json',after);u.write(ROOT/'completion.json',{'base_and_weights_unchanged':True,'raw_valid_candidates':list(results),'time_candidate':selected,'test_candidates':[selected] if selected else [],'reference':'user-selected base14','independent_generalization_verified':False,'new_training_started':False});print('SESSION_COMPLETE',flush=True)
 with (ROOT/'finalize.log').open('x') as log:subprocess.run([u.PYTHON,'-B',str(ROOT/'finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:u.write(ROOT/'failure.json',{'error':repr(exc),'complete':False});raise
