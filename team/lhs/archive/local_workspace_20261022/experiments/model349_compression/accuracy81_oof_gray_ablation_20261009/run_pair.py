from pathlib import Path
import os,subprocess,json,time
ROOT=Path(__file__).resolve().parent;PY='/workspace/venv-komap/bin/python'
for name in ['accuracy81_oof_zgray_20261009','accuracy81_oof_nogray_20261009']:
 path=ROOT.parent/name
 with (path/'queue.log').open('x') as log:
  child=subprocess.Popen([PY,'-B','-u',str(path/'run_sequence.py')],cwd=path,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
  (ROOT/'active_child.json').write_text(json.dumps({'pid':child.pid,'arm':name,'time':time.time()},indent=2)+'\n')
  status=child.wait()
  assert status==0,(name,status)
(ROOT/'completion.json').write_text(json.dumps({'both_finished':True,'time':time.time()},indent=2)+'\n')
