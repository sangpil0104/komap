import time,json,subprocess,os,fcntl
from pathlib import Path
D=Path(__file__).resolve().parent;S=Path('/home/sangpil/komap/codex_lite13_E_seed5_reduction_20261014');PY='/home/sangpil/miniforge3/envs/komap/bin/python'
lock=(D/'resume_seeds.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 if (D/'failure.json').exists():break
 if (D/'completion.json').exists() and (D/'test_completion.json').exists():break
 time.sleep(10)
receipt=json.loads((S/'priority_pause_receipt.json').read_text())
(S/'priority_resume_started.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'checkpoint_epoch':receipt['completed_epoch'],'next_epoch':receipt['next_epoch'],'reason':'shared encoder priority evaluation finished','time':time.time()},indent=2))
with (S/'controller_after_shared_priority.log').open('a') as log:
 subprocess.run([PY,'-u',str(S/'run.py')],cwd=S,stdout=log,stderr=subprocess.STDOUT,check=True)
(D/'seed_resume_completion.json').write_text(json.dumps({'normal_seed_queue_exit':True,'time':time.time()}))
