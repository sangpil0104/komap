import os,time,json,subprocess,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
PY='/workspace/venv-komap/bin/python'
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
assert not (R/'waiter_started.json').exists()
write('waiter_started.json',{'pid':os.getpid(),'time':time.time()})
while True:
 hs=[];free=False
 try:
  for p in sorted(set(R.parent.glob('*/gpu0.lock'))|{R/'gpu0.lock'}):
   h=p.open('a');hs.append(h);fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  free=mem<=2048 and util==0
  write('waiting.json',{'time':time.time(),'locks_free':True,'memory_mib':mem,'util':util})
 except BlockingIOError:write('waiting.json',{'time':time.time(),'locks_free':False})
 finally:
  for h in hs:h.close()
 if free:break
 time.sleep(15)
with (R/'controller.log').open('x') as log:
 c=subprocess.Popen([PY,'-B','-u',str(R/'run_sequence.py')],stdout=log,stderr=subprocess.STDOUT)
 write('controller_started.json',{'pid':c.pid,'time':time.time()})
 rc=c.wait()
write('waiter_completion.json',{'exit_code':rc,'time':time.time()})
