import json,os,signal,time
from pathlib import Path
from tqdm import tqdm
run=Path('/Data3/sppark/komap/experiments/single_unet_v1/outputs/runs/S0_x1_seed2026_official_20260924T045652_436262Z')
worker,parent,pgid=3295633,3295459,3295459
assert os.getpgid(worker)==pgid and os.getpgid(parent)==pgid
assert str(run).encode() in Path(f'/proc/{worker}/cmdline').read_bytes()
output=Path(__file__).parent/'pause_status.json'
progress=tqdm(total=2000,initial=1000,desc='Wait for safe checkpoint',unit='update',dynamic_ncols=True)
last=1000
while True:
 if not Path(f'/proc/{worker}').exists():
  raise RuntimeError('Training worker exited before checkpoint was captured')
 latest=json.loads((run/'latest.json').read_text())
 if latest['update']>=2000:
  checkpoint=run/latest['checkpoint']
  if not checkpoint.is_file():
   raise RuntimeError('Checkpoint marker without checkpoint file')
  os.killpg(pgid,signal.SIGSTOP)
  state=dict(status='paused_after_saved_checkpoint',run=str(run),worker_pid=worker,parent_pid=parent,pgid=pgid,checkpoint=str(checkpoint),update=latest['update'],checkpoint_bytes=checkpoint.stat().st_size)
  output.write_text(json.dumps(state,indent=2)+'\n')
  progress.update(2000-progress.n)
  print('\n'+json.dumps(state),flush=True)
  break
 try:
  lines=(run/'training.jsonl').read_text().splitlines()
  update=json.loads(lines[-1])['update']
  if update>last:
   progress.update(update-last)
   last=update
 except (IndexError,json.JSONDecodeError):
  pass
 time.sleep(.1)
progress.close()
