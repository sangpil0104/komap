"""Wait for the existing controller, then score and report. Never launches training."""
import os, json, time, subprocess, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 guard=ROOT/'finalizer_started.json'
 with guard.open('x') as f: json.dump({'pid':os.getpid(),'time':time.time(),'observed_controller':2309846},f)
 while not (ROOT/'completion.json').exists():
  process=Path('/proc/2309846/cmdline')
  assert process.exists() and b'accuracy81_HR_texture_pair_20261009/run_sequence.py' in process.read_bytes(), 'Controller stopped before completion; inspect existing outputs'
  assert not (ROOT/'controller_failure.json').exists(), 'Controller failure; no retry'
  time.sleep(15)
 subprocess.run(['/workspace/venv-komap/bin/python','-B',str(ROOT/'verify_results.py')],check=True)
 r=json.loads((ROOT/'independent_comparison.json').read_text())
 a=r['arms'];delta=r['texture_minus_control_pp']
 status='다른 fold 확인 대상' if r['pilot_gate_pass'] else '현재 조건의 확대 실험 미채택'
 report=f'''# HRNet 질감 입력 비교 결과

train70 내부 fold1: 학습 52장, 보류 18장. 동일 체크포인트에서 4 epoch 추가 학습한 마지막 가중치를 비교했다.

| 입력 | FP32 평가기 점수 | FP16 저장 확률 독립 채점 |
|---|---:|---:|
| 원본 RGB | {a['original']['fp32_evaluator_percent']:.6f}% | {a['original']['fp16_independent_percent']:.6f}% |
| 질감 채널 | {a['texture']['fp32_evaluator_percent']:.6f}% | {a['texture']['fp16_independent_percent']:.6f}% |

질감 입력의 독립 채점 변화: {delta:+.6f}%p. 결정: **{status}**.

클래스 변화(Al, Al3Ni, Primary Si, Eutectic Si): {r['class_delta_pp']} %p.

FP32 평가 결과와 FP16 독립 채점은 양자화 때문에 구분했다. 단일 fold 파일럿이며 공식 valid20 또는 OOF90 점수가 아니다. 기존 가중치에서 추가 학습한 결과로, 처음부터 재학습하는 재현성 검증을 대신하지 않는다. 이 결과만으로 81% 달성을 주장하지 않는다. 원본 모델 및 데이터는 변경하지 않았다.
'''
 (ROOT/'report.md').write_text(report)
 (ROOT/'finalizer_completion.json').write_text(json.dumps({'time':time.time(),'independent_scoring_completed':True,'report_created':True,'pilot_gate_pass':r['pilot_gate_pass']},indent=2)+'\n')
if __name__=='__main__':
 try: main()
 except BaseException:
  (ROOT/'finalizer_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n');raise
