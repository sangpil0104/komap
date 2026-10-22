import os,sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_N3_L2_then_resume_queue_20261013';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';L=K/'codex_L2_seed4_20261013';T=K/'codex_N3_L2_seed_retrained_test_20261013';E=K/'codex_r326_ema_seed4_20261012';H=K/'codex_hr392_set5_resume_20261012'
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(stage):dump(O/'status.json',{'stage':stage,'pid':os.getpid(),'time':time.time()})
def wait_done(folder,name='completion.json',failures=('failure.json',)):
 while not (folder/name).exists():
  for f in failures:
   if (folder/f).exists():raise RuntimeError('Preceding stage failure: '+str(folder/f))
  time.sleep(20)
def run(folder):
 status('executing '+folder.name)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1')
 with (folder/'queue_controller.log').open('a') as f:
  child=subprocess.Popen(['flock','-n',str(folder/'run.lock'),sys.executable,'-u',str(folder/'run.py')],stdout=f,stderr=subprocess.STDOUT,env=env);dump(O/'active_child.json',{'pid':child.pid,'experiment':folder.name,'time':time.time()});assert child.wait()==0,'Stage failed or already locked: '+folder.name
 wait_done(folder)
def decision():
 d=json.loads((L/'evaluation/seed361/result.json').read_text());n=json.loads((F/'N3/result.json').read_text());base=json.loads((F/'baseline/result.json').read_text())['percent'];test=json.loads((T/'comparison.json').read_text());one=d['variants']['L2'];both=d['variants']['both'];delta=both['percent']-base
 recommend=delta>0 and both['worst_image_delta_pp']>=-.1 and both['percent']>=max(one['percent'],n['percent'])
 receipt={'N3_seed':414,'L2_seed':361,'baseline':base,'N3_percent':n['percent'],'L2_percent':one['percent'],'both_percent':both['percent'],'both_worst_image_delta_pp':both['worst_image_delta_pp'],'screen_pass_mean_and_local':recommend,'test_GT_used':False,'test_change_percent':{k:v['changed_percent'] for k,v in test['results'].items()},'test_gallery_needs_visual_review':True,'automatic_release_replacement':False,'recommendation':'candidate for visual/fresh confirmation' if recommend else 'defer simultaneous replacement: mean/local criteria not all met','valid_development_score':True};dump(O/'comparison_decision.json',receipt)
 report='N3/L2 새학습 고정시드 교체 비교\n\n'+json.dumps(receipt,ensure_ascii=False,indent=2)+'\n\n동시교체의 점수와 국소손실 조건을 함께 판단. test 변경률은 정확도가 아니며, 갤러리 시각검토 전 채택확정 안함. 릴리스 미교체.\n';(O/'report.md').write_text(report)
def ensure_waiter(pid,script):
 args=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
 assert script in args,'Expected original waiter unavailable; inspect before duplicate start: '+script
 return args.strip()
def main():
 if (O/'completion.json').exists():return
 status('waiting fresh N3 and original L2 validation')
 wait_done(F)
 assert json.loads((F/'result.json').read_text())['results']['baseline']['changed_pixels']==0,'Fresh control mismatch'
 run(L);run(T);decision()
 # Completed set0 models remain untouched. The early interrupted set1/fold1 had no saved state.
 assert not (E/'failure.json').exists(),'Inspect original326 failure before restart'
 for fold in [1,2]:assert (E/f'training/set0/fold{fold}/completion.json').exists()
 args=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 assert not any(str(E/'run.py') in line for line in args.splitlines()),'Original326 controller already exists'
 ensure_waiter(351745,'evaluate_lite16.py');ensure_waiter(432346,'codex_hr392_set5_resume_20261012/run.py')
 incomplete=E/'training/set1/fold1'
 if not (incomplete/'completion.json').exists() and not (incomplete/'train_state.pt').exists():
  archive=incomplete/'interrupted_400steps_before_N3_queue';archive.mkdir(exist_ok=True)
  for name in ['train.log','args.json']:
   p=incomplete/name
   if p.exists():assert not (archive/name).exists(),'Archive already exists; inspect restart';p.rename(archive/name)
 dump(E/'user_resume_after_L2_20261013.json',{'authorized':True,'after':T.name,'time':time.time(),'preserved_completed_jobs':2,'remaining_jobs':6,'original_eval_waiter':351745,'original_HR_waiter':432346})
 run(E)
 status('waiting existing326 lite16 evaluator')
 wait_done(E,'lite16_evaluation_completion.json',('failure.json','lite16_evaluation_failure.json'))
 status('waiting existing HR392 set5 resume and evaluation')
 wait_done(H)
 dump(O/'completion.json',{'time':time.time(),'L2_four_seed_done':True,'retrained_N3_L2_test_done':True,'R326_remaining_and_lite16_eval_done':True,'HR392_set5_train_and_eval_done':True,'release_replaced':False});status('complete')
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'failure.json',{'time':time.time(),'trace':traceback.format_exc()});status('failed: inspect failure.json');raise
