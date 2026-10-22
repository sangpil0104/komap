import os,sys,subprocess,json,time,signal,traceback
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_lite16_N3_L2_test_20261012';ticket=json.loads((O/'priority_authorized.json').read_text());pid=ticket['paused_controller_pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
assert 'codex_r326_ema_seed4_20261012/run.py' in cmd,'Controller identity changed'
code=subprocess.call([sys.executable,'-u',str(O/'run.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'))
if code==0 and (O/'completion.json').exists():
 assert 'codex_r326_ema_seed4_20261012/run.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
 os.kill(pid,signal.SIGCONT);(O/'training_resumed.json').write_text(json.dumps({'pid':pid,'time':time.time(),'reason':'priority test measurement complete; GPU child exited'}))
else:
 (O/'priority_failure.json').write_text(json.dumps({'exit_code':code,'paused_controller_pid':pid,'time':time.time(),'reason':'test comparison failed; training controller remains paused for inspection'}));sys.exit(code or 1)
