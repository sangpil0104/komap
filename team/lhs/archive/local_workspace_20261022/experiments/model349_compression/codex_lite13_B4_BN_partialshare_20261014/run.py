import os,sys,json,time,hashlib,traceback,fcntl,importlib.util,gc,re,subprocess
from pathlib import Path
import numpy as np,torch
D=Path(__file__).resolve().parent;K=Path('/home/sangpil/komap')
sp=importlib.util.spec_from_file_location('shared',K/'codex_lite13_shared_encoder_20261014/run.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def dump(name,data):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def load_members():return {tag:(s.rmod if tag.startswith('R') else s.emod).load_member(str(folder),'cuda','last.pt') for tag,folder in s.folders.items()}
def apply(members,tags,mode):
 anchor=members[tags[0]]['model'].encoder;conv_count=0;bn_exact=True
 for tag in tags[1:]:
  enc=members[tag]['model'].encoder
  if mode=='BN':
   before={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in enc.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
   source=dict(anchor.named_modules())
   for name,m in enc.named_modules():
    if not isinstance(m,torch.nn.Conv2d):continue
    src=source[name];assert type(m)==type(src) and m.weight.shape==src.weight.shape
    m.weight=src.weight;m.bias=src.bias;assert m.weight.data_ptr()==src.weight.data_ptr();conv_count+=1
   for name,m in enc.named_modules():
    if name in before:assert all(torch.equal(v.detach().cpu(),before[name][k]) for k,v in m.state_dict().items()),(tag,name)
  else:
   n=int(mode);assert hasattr(enc,'_blocks') and len(enc._blocks)>=n
   enc._conv_stem=anchor._conv_stem;enc._bn0=anchor._bn0
   for i in range(n):enc._blocks[i]=anchor._blocks[i]
  members[tag]['model'].eval()
 return {'mode':mode,'anchor':tags[0],'heads':tags,'shared_conv_parameter_objects':conv_count,'BN_preserved_exact':mode=='BN','prefix_blocks_shared':0 if mode=='BN' else int(mode)}
def export(case,members,tags):
 folder=D/'packages'/case;folder.mkdir(parents=True,exist_ok=True);bank={};mapping={};seen={}
 for tag in tags:
  refs={}
  for key,v in members[tag]['model'].state_dict().items():
   identity=(v.data_ptr(),tuple(v.shape),str(v.dtype),tuple(v.stride()))
   if identity not in seen:
    token=f't{len(bank):04d}';seen[identity]=token;bank[token]=v.detach().cpu().clone()
   refs[key]=seen[identity]
  mapping[tag]=refs
 path=folder/'tensors.pt';torch.save(bank,path)
 raw=sum(v.numel()*v.element_size() for v in bank.values());before=sum(sum(v.numel()*v.element_size() for v in members[tag]['model'].state_dict().values()) for tag in tags)
 info={'tensor_bytes':raw,'unshared_tensor_bytes':before,'tensor_saved_bytes':before-raw,'physical_checkpoint_bytes':path.stat().st_size,'sha256':s.sha(path),'output_heads':len(tags),'state_refs':mapping,'full_encoder_forward_per_member':True,'actual_shared_storage_not_just_file_renaming':True}
 dump(f'packages/{case}/manifest.json',info);return info

def run():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 frozen=json.loads((K/'codex_lite13_shared_encoder_20261014/preflight.json').read_text())['source_sha'];assert all(s.sha(p)==h for p,h in frozen.items());dump('source_sha.json',frozen)
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  dump('status.json',{'phase':'waiting_GPU_idle','pid':os.getpid()});time.sleep(5)
 images=sorted((K/'data/valid/images').glob('*_image.png'));assert len(images)==20;inputs={str(p):s.sha(p) for p in images};dump('input_sha.json',inputs)
 jobs=[('BN_R','BN',['R1','R3','R4']),('BN_E','BN',['E1','E3']),('BN_ALL','BN',['R1','R3','R4','E1','E3'])]+[(f'prefix{n}_{g}',str(n),['R1','R3','R4'] if g=='R' else ['E1','E3']) for n in (4,8,16) for g in ('R','E')]
 dump('planned_jobs.json',jobs);timings={};packages={}
 for case,mode,tags in jobs:
  o=D/'probabilities'/case;o.mkdir(parents=True,exist_ok=True)
  if (o/'completion.json').exists():continue
  dump('status.json',{'phase':'GPU inference','case':case,'pid':os.getpid()});members=load_members();proof=apply(members,tags,mode);packages[case]=export(case,members,tags);dump(f'probabilities/{case}/sharing_proof.json',proof)
  count={'calls':0}
  def hook(mod,args,output):count['calls']+=1
  handles=[members[tag]['model'].encoder.register_forward_hook(hook) for tag in tags]
  start=time.time();torch.cuda.reset_peak_memory_stats();rows=[]
  for image in images:
   name=image.name.removesuffix('_image.png');roles=[g for g in ('R','E') if any(t.startswith(g) for t in tags)]
   if all((o/f'{name}_{g}.npy').exists() for g in roles):continue
   count['calls']=0;tick=time.time();gray=s.rmod.load_gray(image)
   for role in roles:
    selected=[members[tag] for tag in tags if tag.startswith(role)];mod=s.rmod if role=='R' else s.emod
    p=mod.predict_probs(selected,gray,tta=role=='E',device='cuda').cpu().numpy().astype(np.float32);assert np.isfinite(p).all();np.save(o/f'{name}_{role}.npy',p)
   expected=sum(1 if tag.startswith('R') else 8 for tag in tags);assert count['calls']==expected,(case,count,expected)
   rows.append({'image':name,'seconds':time.time()-tick,'actual_encoder_calls':count['calls']});print(case,name,count['calls'],flush=True)
  for handle in handles:handle.remove()
  timings[case]={'seconds_this_attempt':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rows':rows};dump(f'probabilities/{case}/completion.json',timings[case]);del members;gc.collect();torch.cuda.empty_cache()
 assert all(s.sha(p)==h for p,h in inputs.items());assert all(s.sha(p)==h for p,h in frozen.items());dump('gpu_completion.json',{'timings':timings,'source_sha_unchanged':True,'input_sha_unchanged':True,'test_GT_used':False})
 dump('status.json',{'phase':'CPU scoring','pid':os.getpid()});subprocess.run([sys.executable,str(D/'score.py')],check=True)
 dump('completion.json',{'finished':time.time(),'new_training_used':False,'seeds_remain_paused':True,'release_replaced':False,'test_GT_used':False});dump('status.json',{'phase':'complete','seeds_remain_paused':True})

if __name__=='__main__':
 lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (D/'completion.json').exists():sys.exit(0)
 try:run()
 except Exception:dump('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
