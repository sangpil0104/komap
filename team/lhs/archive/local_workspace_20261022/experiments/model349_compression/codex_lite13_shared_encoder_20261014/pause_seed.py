from pathlib import Path
import os,json,time,signal,shutil,torch
D=Path(__file__).resolve().parent;S=Path('/home/sangpil/komap/codex_lite13_E_seed5_reduction_20261014');OUT=S/'training/hr_seed9001';parent=json.loads((S/'status.json').read_text())['pid']
assert Path(f'/proc/{parent}/cwd').resolve()==S
cmd=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'run.py' in cmd
children=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  data=(p/'status').read_text();ppid=int(next(x.split()[1] for x in data.splitlines() if x.startswith('PPid:')))
  if ppid==parent:children.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(children)==1,children;child=children[0];cmd=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(OUT) in cmd and 'model392/train.py' in cmd
old=(OUT/'train_state.pt').stat().st_mtime_ns
os.kill(parent,signal.SIGSTOP)
# Wait for the next completed checkpoint, never stop during serialization.
while True:
 state=OUT/'train_state.pt'
 try:
  mt=state.stat().st_mtime_ns
  opened=[os.readlink(p) for p in Path(f'/proc/{child}/fd').iterdir()]
  writing=any(str(OUT/'train_state.pt') in p or str(OUT/'last.pt') in p for p in opened)
  if mt!=old and not writing and time.time_ns()-mt>300_000_000:break
 except FileNotFoundError:pass
 time.sleep(.15)
os.kill(child,signal.SIGSTOP)
try:
 state=torch.load(OUT/'train_state.pt',map_location='cpu',weights_only=False);last=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
 model=last.get('state_dict',last.get('model_state_dict',last.get('model')))
 assert isinstance(model,dict),last.keys()
 assert set(model)==set(state['model'])
 assert all(torch.equal(model[k],state['model'][k]) for k in model),'last.pt/train_state.pt inconsistent'
 epoch=int(state['epoch']);backup=S/f'priority_pause_epoch{epoch}';backup.mkdir(exist_ok=True)
 for name in ('last.pt','train_state.pt','args.json','train.log'):shutil.copy2(OUT/name,backup/name)
 receipt={'parent':parent,'child':child,'completed_epoch':epoch,'next_epoch':epoch+1,'output':str(OUT),'backup':str(backup),'last_and_train_state_model_exact':True,'resume_authorized_by_user_priority':True,'time':time.time()}
 (S/'priority_pause_receipt.json').write_text(json.dumps(receipt,indent=2));(OUT/'resume_authorized.json').write_text(json.dumps(receipt,indent=2));print('CHECKPOINT_PRESERVED',epoch,flush=True)
 # Only descendants of our verified trainer belong to this job.
 descendants=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   ppid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')))
   if ppid==child:descendants.append(int(p.name))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 for pid in descendants+[child,parent]:
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 (S/'status.json').write_text(json.dumps({'phase':'paused_for_shared_encoder_priority','completed_epoch':epoch,'resume_from_epoch':epoch+1,'old_controller':parent},indent=2));print('OWN_SEED_GPU_RELEASED',child,flush=True)
except Exception:
 os.kill(child,signal.SIGCONT);os.kill(parent,signal.SIGCONT);raise
