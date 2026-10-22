import argparse,fcntl,os,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
child=None
def stop(signum,frame):
 if child is not None and child.poll() is None:
  child.terminate(); child.wait()
 raise SystemExit(128+signum)
for sig in [signal.SIGINT,signal.SIGTERM]: signal.signal(sig,stop)
def call(argv):
 global child
 child=subprocess.Popen(argv)
 code=child.wait()
 if code: raise SystemExit(code)
p=argparse.ArgumentParser(); p.add_argument('command',choices=['check','train'],default='check',nargs='?'); p.add_argument('--execute',action='store_true'); p.add_argument('--model',choices=['all','349','397'],default='all'); args=p.parse_args()
(ROOT/'outputs').mkdir(exist_ok=True)
with (ROOT/'outputs/launcher.lock').open('a+') as lock:
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: raise SystemExit('Another launcher is active')
 execute=args.command=='train' and args.execute
 for model in (['349','397'] if args.model=='all' else [args.model]):
  command=[sys.executable,'-B','-u',str(ROOT/f'model{model}/run.py'),'train' if execute else 'check']
  if execute: command+=['--execute']
  if model=='349': command+=['--gpus','4,5,6,7']
  print(f'Model{model}: {"TRAIN" if execute else "CHECK (no training)"}',flush=True)
  call(command)
