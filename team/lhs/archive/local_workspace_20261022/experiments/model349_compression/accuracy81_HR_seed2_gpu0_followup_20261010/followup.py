"""One-shot continuation; never relaunches existing GPU stages."""
import os,time,json,subprocess,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PY='/workspace/venv-komap/bin/python'
def write(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
 with (ROOT/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'recovery_source':'epoch1 checkpoint','time':time.time()},f)
 for fold in [3,4]:
  work=BASE/('accuracy81_HR_seed2_warmstart_fold3_gpu0_20261010' if fold==3 else f'accuracy81_HR_seed2_warmstart_fold{fold}_20261010')
  assert not (work/'waiter_started.json').exists(), 'Existing stage; inspect without relaunch'
  write('status.json',{'phase':'waiting_or_running_fold','fold':fold,'time':time.time()})
  subprocess.run([PY,'-B','-u',str(work/'wait_for_gpu.py')],check=True)
  subprocess.run([PY,'-B','-u',str(work/'finalize.py')],check=True)
  assert json.loads((work/'completion.json').read_text())['inputs_unchanged']
  e=BASE/f'accuracy81_HR_seed2_E_fold{fold}_20261010'
  for script,out in [('screen.py','result.json'),('both_screen.py','both_result.json')]:
   assert not (e/out).exists(), 'Existing CPU output: inspect, do not overwrite'
   subprocess.run([PY,'-B','-u',str(e/script)],check=True)
 records=[];both=[];folds=[]
 for f in range(1,5):
  e=BASE/f'accuracy81_HR_seed2_E_fold{f}_20261010'
  a=json.loads((e/'result.json').read_text());b=json.loads((e/'both_result.json').read_text())
  records+=a['records'];both+=b['records'];folds.append(a['E_delta_pp'])
 assert len(records)==70 and len({r['name'] for r in records})==70
 groups={}
 for r in records:
  g=r['name'].split('_')[0];assert groups.get(g,r['fold'])==r['fold'];groups[g]=r['fold']
 assert len(groups)==59
 means={k:float(100*np.mean([r[k] for r in records])) for k in ['E_before','E_after','HR_before','HR_after']}
 bm={k:float(100*np.mean([r[k] for r in both])) for k in ['E_before','E_after']}
 delta=means['E_after']-means['E_before'];gate=delta>0 and sum(v>0 for v in folds)>=3
 write('held70_gate.json',{'expand_valid20':gate,'means_percent':means,'E_delta_pp':delta,'fold_E_delta_pp':folds,'both_means_percent':bm,'both_delta_pp':bm['E_after']-bm['E_before'],'records':records,'both_records':both,'scope':'train70 held70 only; recipe developed on held folds; not unbiased generalization'})
 if not gate:
  write('completion.json',{'held70_completed':True,'valid20_started':False,'reason':'fixed expansion gate not met','time':time.time()});return
 valid=BASE/'accuracy81_HR_seed2_valid20_20261010'
 write('status.json',{'phase':'valid20_fixed_candidate','time':time.time()})
 subprocess.run([PY,'-B','-u',str(valid/'wait_for_gpu.py')],check=True)
 subprocess.run([PY,'-B','-u',str(valid/'compare.py')],check=True)
 write('completion.json',{'held70_completed':True,'valid20_completed':True,'release_replaced':False,'time':time.time()})
 write('status.json',{'phase':'completed','time':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  write('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
