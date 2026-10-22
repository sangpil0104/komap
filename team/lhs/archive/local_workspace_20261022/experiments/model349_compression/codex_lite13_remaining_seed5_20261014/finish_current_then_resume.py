import json,time,os,signal,hashlib,subprocess,traceback
from pathlib import Path
D=Path(__file__).resolve().parent
try:
 t=json.load(open(D/'queue_transition.json'));parent=t['controller'];child=t['child'];out=Path(t['out'])
 while True:
  stat=Path(f'/proc/{child}/stat').read_text();fields=stat.rsplit(') ',1)[1].split()
  if fields[0]=='Z':
   assert int(fields[49])==0,fields[49]
   break
  time.sleep(2)
 assert (out/'summary.json').exists() and (out/'best.pt').exists()
 h=hashlib.sha256()
 with (out/'best.pt').open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 (out/'training_completion.json').write_text(json.dumps({'normal_exit':True,'checkpoint_sha':h.hexdigest(),'finished':time.time(),'receipt_recovered_after_user_queue_change':True},indent=2))
 assert Path(f'/proc/{parent}/cwd').resolve()==D
 assert b'run.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
 os.kill(parent,signal.SIGKILL)
 time.sleep(1)
 with (D/'controller_after_R329_cancel.log').open('a') as log:
  p=subprocess.Popen(['/home/sangpil/miniforge3/envs/komap/bin/python','-u',str(D/'run.py')],cwd=D,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 t.update(state='resumed without R329',new_controller=p.pid,finished=time.time())
 (D/'queue_transition.json').write_text(json.dumps(t,indent=2))
except Exception:
 (D/'queue_transition_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
