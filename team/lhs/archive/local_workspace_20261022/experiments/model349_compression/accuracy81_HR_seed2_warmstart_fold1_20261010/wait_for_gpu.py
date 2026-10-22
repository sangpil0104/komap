import os,json,time,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
try:
 with (ROOT/'waiter_started.json').open('x') as f:json.dump({'pid':os.getpid()},f)
 while True:
  assert not (ROOT/'started.json').exists()
  apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
  (ROOT/'waiter_status.json').write_text(json.dumps({'time':time.time(),'apps':apps,'memoryMiB':mem,'util':util}))
  if not apps and mem<=64 and util==0:break
  time.sleep(30)
 subprocess.run([sys.executable,'-B','-u',str(ROOT/'run_sequence.py')],check=True)
except BaseException:
 (ROOT/'waiter_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
