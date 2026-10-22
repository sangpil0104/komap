import argparse,fcntl,json,os,shutil,subprocess,sys,hashlib
from datetime import datetime,timezone
from pathlib import Path
from scheduler import run_tasks, validate_tasks
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('command',choices=['check','train']); p.add_argument('--execute',action='store_true'); args=p.parse_args()
 if Path(sys.prefix).name!='komap': raise RuntimeError('conda komap required')
 for name,h in json.loads((ROOT/'source_manifest.json').read_text()).items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h: raise ValueError(name)
 if args.command=='check' or not args.execute:
  subprocess.run([sys.executable,str(ROOT/'source/scripts/smoke_test.py')],check=True)
  print('Model397 CPU smoke OK; preparing real-data execution plan')
 (ROOT/'outputs').mkdir(exist_ok=True)
 with (ROOT/'outputs/pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  execute=args.command=='train' and args.execute
  tag='run_' if execute else 'preflight_'
  run=ROOT/'outputs'/datetime.now(timezone.utc).strftime(tag+'%Y%m%dT%H%M%S_%fZ'); run.mkdir()
  shutil.copytree(ROOT/'source',run/'source_snapshot'); source=run/'source_snapshot'
  for f in ['run.py','scheduler.py','fold.sh','worker.py','export_valid.py','source_manifest.json']: shutil.copy2(ROOT/f,run/f)
  # fold.sh derives its bundle location from the worker directory.
  (run/'source').symlink_to(source,target_is_directory=True)
  (run/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
  subprocess.run([sys.executable,str(source/'scripts/prepare_oof_splits.py'),'--data-root',str(REPO/'dataset'),'--output-root',str(run/'splits')],check=True)
  # B4 predictor expects an image directory for the chosen split.
  (run/'splits/model329_cv/valid').symlink_to(REPO/'dataset/valid',target_is_directory=True)
  tasks=[]
  def add(name,argv,deps=(),kind='gpu',produces=()):
   tasks.append(dict(id=name,argv=argv,deps=list(deps),kind=kind,priority=len(tasks),serial_index=len(tasks),produces=list(produces)))
  oof=[]; valid=[]
  for family,seeds in [('b4',[0]),('hr',[1,2])]:
   for seed in seeds:
    for fold in range(1,5):
     name=f'{family}_s{seed}_f{fold}'; oof.append(name)
     add(name,[str(run/'worker.py'),str(REPO/'dataset'),str(run),str(fold),family,str(seed)],produces=[str(run/(f'model329/fold{fold}_oof/probabilities.npz' if family=='b4' else f'model392/seed{seed}/fold{fold}_oof/valid_probabilities.npz'))])
     out=run/'valid'/name; valid.append(name+'_valid')
     if family=='b4':
      argv=[str(source/'model329/predict.py'),'--data',str(run/'splits/model329_cv'),'--split','valid','--runs',str(run/f'model329/fold{fold}'),'--out',str(out),'--save_probs']
     else:
      argv=[str(source/'model392/evaluate_image_mean.py'),'--checkpoint',str(run/f'model392/seed{seed}/fold{fold}/last.pt'),'--data-root',str(REPO/'dataset'),'--split','valid','--output-dir',str(out),'--device','cuda:0','--batch-size','2','--patch-size','512','--overlap','170','--input-scale','2.0','--tta-mode','flip4']
     add(name+'_valid',argv,[name],produces=[str(out/('probabilities.npz' if family=='b4' else 'valid_probabilities.npz'))])
  add('oof_score',[str(source/'scripts/score_model397_oof.py'),'--data-root',str(REPO/'dataset'),'--output-root',str(run)],oof,'cpu')
  add('valid_export',[str(run/'export_valid.py'),str(run),str(REPO)],valid+['oof_score'],'cpu')
  (run/'tasks.json').write_text(json.dumps(tasks,indent=2))
  inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
  devices={int(a.strip()):b.strip() for a,b in (line.split(',') for line in inventory.splitlines())}
  env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['PYTHONDONTWRITEBYTECODE']='1'
  print(f'RUN_DIRECTORY={run}',flush=True)
  validate_tasks(tasks)
  if not all(g in devices for g in [4,5,6,7]): raise RuntimeError('GPUs 4,5,6,7 required')
  if not execute:
   print(f'Model397 CHECK OK: real 70-image splits, 12 training/OOF + 12 valid + 2 CPU tasks, GPU 4,5,6,7; no training')
   return
  run_tasks(tasks,run,source,[4,5,6,7],devices,sys.executable,env,lock_fd=lock.fileno())
if __name__=='__main__': main()
