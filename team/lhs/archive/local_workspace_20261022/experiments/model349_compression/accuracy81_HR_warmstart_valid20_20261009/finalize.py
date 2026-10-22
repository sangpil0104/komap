import os,json,time,subprocess,traceback,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTROLLER=2325188
def main():
 with (ROOT/'finalizer_started.json').open('x') as f:json.dump({'pid':os.getpid(),'controller':CONTROLLER},f)
 while not (ROOT/'completion.json').exists():
  proc=Path(f'/proc/{CONTROLLER}/cmdline')
  assert proc.exists() and b'accuracy81_HR_warmstart_valid20_20261009/run_sequence.py' in proc.read_bytes(), 'Controller ended without completion; inspect, do not restart'
  assert not (ROOT/'controller_failure.json').exists()
  time.sleep(15)
 subprocess.run([sys.executable,'-B','-u',str(ROOT/'compare.py')],check=True)
 r=json.loads((ROOT/'comparison.json').read_text())
 (ROOT/'report.md').write_text(f'# HR 추가 학습 valid20 고정 결합 비교\n\n기존 {r["before_percent"]:.6f}%, 새 후보 {r["after_percent"]:.6f}%, 변화 {r["delta_pp"]:+.6f}%p.\n\n비율과 Ni·Al 덮어쓰기 규칙을 고정했다. valid20 반복 사용의 선택 편향이 남는다. 새 HR은 fold별 FP16 저장 후 평균하므로 최종 채택 전 FP32 fold 평균 fresh 추론을 확인해야 한다. 현재 결과만으로 독립 일반화·81% 달성을 주장하지 않는다.\n')
 (ROOT/'finalizer_completion.json').write_text(json.dumps({'scoring_completed':True,'delta_pp':r['delta_pp'],'time':time.time()},indent=2)+'\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'finalizer_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n');raise
