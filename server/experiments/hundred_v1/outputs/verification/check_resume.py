"""Run after the smoke suite: pause, resume, evaluate, predict, and reuse."""
import sys,time,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
name='m016_aspp_w64_staged'
status_path=ROOT/'outputs/smoke_suite_status.json'
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    if status_path.exists():
        state=json.loads(status_path.read_text())
        if state['state']!='running':
            assert not state['failures'],state
            assert len(state['completed'])==14,state
            break
    time.sleep(3)
else:raise TimeoutError('Smoke suite did not finish')
command=[sys.executable,'-B',str(ROOT/'run.py')]
run=ROOT/'outputs/smoke_runs'/name
log=ROOT/'outputs/verification/resume.log'
with log.open('w') as out:
    def execute(*args):subprocess.run(command+list(args),stdout=out,stderr=subprocess.STDOUT,check=True)
    execute('train',name,'--gpus','8','--smoke','--stop-after','1')
    paused=json.loads((run/'status.json').read_text())
    assert paused['state']=='paused' and paused['update']==1
    execute('train',name,'--gpus','8','--smoke','--resume')
    completed=json.loads((run/'status.json').read_text())
    assert completed['state']=='completed' and completed['update']==4
    execute('evaluate',name,'--smoke')
    execute('evaluate',name,'--smoke','--d4')
    execute('predict',name,'--smoke','--d4')
    before=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()
    execute('suite','--gpus','8','--smoke','--experiments',name)
    after=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()
    assert before==after
    masks=json.loads((ROOT/'outputs/smoke_predictions'/name/'d4/check.json').read_text())
    assert masks['images']==10
text=log.read_text()
assert '이미 학습 완료' in text and '평가 완료 결과 재사용' in text
result=dict(paused_at=1,resumed_to=4,world_size=8,completed_training_reused=True,
            completed_evaluation_reused=True,test_masks_checked=10)
(ROOT/'outputs/verification/resume_audit.json').write_text(json.dumps(result,indent=2))
subprocess.run([sys.executable,'-B',str(ROOT/'outputs/verification/check_pipeline.py')],check=True)
print('Pause/resume, evaluation reuse, and test masks: PASS')
