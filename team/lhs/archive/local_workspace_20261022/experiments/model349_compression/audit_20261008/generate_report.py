"""Generate Korean summary strictly from verified audit files."""
import json
from pathlib import Path
ROOT=Path('/workspace/model349_compress/audit_20261008')
def load(p):return json.loads(p.read_text())
report=load(ROOT/'comparison.json'); inv=load(ROOT/'checkpoint_inventory.json'); batch=load(ROOT/'batched_summary.json'); extra=load(ROOT/'hramp_summary.json') if (ROOT/'hramp_summary.json').exists() else {}
accepted=batch.get('exploratory_adoption_pass') or extra.get('exploratory_adoption_pass')
lines=['# Model349 경량화 재개 감사 결과 (2026-10-08)','',
'## 판정','',
('HRNet FP16 autocast와 tile batch 2→8을 포함한 추가 후보가 사전 탐색 채택 기준을 통과했다. 두 변경을 포함한 파이프라인 전체의 효과이며 각각의 효과로 분리하지 않는다.' if extra.get('exploratory_adoption_pass') else ''),
('15개 모델을 모두 보존한 **FP16 저장 + OpenCV 2 threads + 고해상도 D4 view batch4**가 사전 탐색 채택 기준을 통과했다.' if batch.get('exploratory_adoption_pass') else 'D4 배치 후보는 사전 탐색 채택 기준을 통과하지 못했다. 저장 크기 개선과 성능·시간 판정은 구분한다.'),'',
'valid20 원본 영상 재추론과 반복 시간의 재현 결과다. 이 valid를 반복 사용했으므로 독립 일반화 검증이나 test 점수 유지 근거로 해석하지 않는다. 기존 릴리스·원본 가중치·미완료 산출물은 덮어쓰지 않았다.','',
'## 감사와 실행 조건','',
'- GPU 사용자 매뉴얼 전체를 읽었다. GPU6은 타 컨테이너의 작업(PID 549152, 3328 MiB)이 점유하여 사용하지 않았다. 비어 있던 P100 GPU7 한 장에서 기준과 후보를 순차 실행했다.','- GPU6 지정 기존 프로토콜을 GPU7의 동시기 기준 비교로 보완했으며, 결과 전 문서에 기록했다. flock·현재 프로세스 확인·GPU 점유 재확인으로 중복을 막았다. 다른 사용자 작업은 조작하지 않았다.','- 과거 FP16 TENT5는 5개 component 모두 20장·100개 확률을 완료했고 fusion에서 마스크 1장만 남아 있었다. 별도 경로로 CPU fusion을 복구했다(80.0655029%). 전체 시간은 알 수 없으므로 null이며 속도 근거에서 제외한다. 중단 원인은 로그만으로 확정하지 못했다.','- 과거 TENT3/12에는 완료·부분 산출물이 없었다. 새 경로에 원본 영상부터 실행했다. 이전 고정 PID 대기 큐는 재사용하지 않았다.','- 초기 lite15는 614.73초였지만 CPU load와 width4000 B4 지연이 컸다. 이 실행과 과거 366.56초 단회만으로 후보 속도를 판단하지 않는다.','',
'## 공식 valid20 재추론','',
'지표는 이미지별 네 클래스 IoU 평균을 계산한 뒤 20장 평균이다. 정답에 없는 클래스도 IoU=0이며 empty union=0이다. 별도 채점 코드가 label PNG와 GT를 다시 읽어 metrics.json과 1e-12 이내 일치를 확인했다. 입력 영상과 calibration의 SHA-256도 동일함을 확인했다.','',
'| 실행 | 가중치 수 | 용량(GB, 10⁹ bytes) | image mIoU | 이번 lite15 대비 | 최악 이미지 변화 | 시간(초) | 품질 기준 |',
'|---|---:|---:|---:|---:|---:|---:|---|']
keys=['historical349','historical15','historical14','lite15','cpu2_lite15','cpu2_fp16_t5','cpu2_fp16_t3','cpu2_fp16_12','cpu2_fp16_10','fp16_15_batched','fp16_15_batched_repeat2','fp16_15_batched_repeat3','fp16_15_batched_hramp','fp16_15_batched_hramp_repeat2','fp16_15_batched_hramp_repeat3']
for key in keys:
 if key not in report['runs']:continue
 row=report['runs'][key]; q=row['vs_same_gpu_lite15']; cp=row.get('checkpoints')
 if cp is None: cp=inv.get({'historical349':'original349','historical15':'lite15','historical14':'lite14_fp32'}.get(key,''))
 size=f"{cp['bytes']/1e9:.6f}" if cp else '—'; count=str(cp['count']) if cp else '—'
 seconds=row['completion']['elapsed_seconds']; t=f'{seconds:.2f}' if seconds is not None else '복구, 전체 시간 불명'
 quality_label='참고/기준' if key in ('historical349','historical15','historical14','lite15','cpu2_lite15') else ('통과' if q['quality_pass'] else '실패')
 lines.append(f"| {key} | {count} | {size} | {row['miou_percent']:.7f}% | {q['delta_pp']:+.7f}%p | {q['worst_image_delta_pp']:+.7f}%p | {t} | {quality_label} |")
lines+=['','historical 행의 시간은 과거 단회 기록이며 이번 반복 시간과 직접 속도를 비교하지 않는다. 초기 후보 시간도 단회 진단이다.','',
'- FP16 14개 저장판의 품질은 거의 유지되지만 단회 속도 개선은 확인되지 않았다. FP16 저장은 기존 FP32 module로 로드하므로 계산/VRAM 절감으로 주장하지 않는다.','- TENT3는 평균 -0.0157446%p, A11 최악 -0.2007161%p로 평균 0.01%p·이미지 0.10%p 손실 한도를 모두 초과해 기각한다.','- 12개 fold326 1·3은 품질 기준을 통과했지만 valid에서 고른 fold 탐색이다. 단회 약 5% 시간 차이는 속도 확정 근거가 아니며 기본판으로 교체하지 않는다.','- 10개는 326의 모든 가중치와 forward를 실제 제거하고 gamma329 Al3Ni=0.8로 고정했다. 추가 계수 선택이나 fold 탐색을 하지 않았다. 원본 재추론 평균은 80.1188236%지만 A153 최악 -0.2232411%p로 이미지 한도 0.10%p를 초과해 기각한다.','',
'## 클래스별 변화','',
'아래 변화는 이번 lite15 FP32 원본 재추론 대비이며, 각 클래스의 20-image mean IoU 차이다.','',
'| 실행 | Al3Ni | Primary Si | Al | Eutectic Si |','|---|---:|---:|---:|---:|']
for key in keys[4:]:
 if key in report['runs']:
  vals=report['runs'][key]['vs_same_gpu_lite15']['class_delta_pp'];lines.append('| '+key+' | '+' | '.join(f'{v:+.7f}%p' for v in vals)+' |')
lines+=['','전체 이미지별 mIoU·4개 클래스 변화, 마스크 변화 픽셀은 [comparison.json](comparison.json)에 보존했다.','']
lines+=['## 보관된 원본 349·lite14 마스크 대비','', '입력과 calibration SHA가 일치하는 기존 공식 산출물에서 다시 채점한 비교다. 아래 시간 비교는 하지 않으며, 사전 채택 판정 기준은 이번 lite15 및 각 동시기 기준 실행이다.','', '| 실행 | 원본349 평균 변화 | 원본349 최악 이미지 변화 | lite14 평균 변화 | lite14 최악 이미지 변화 |','|---|---:|---:|---:|---:|']
for key in keys[5:]:
 if key not in report['runs']:continue
 row=report['runs'][key];a=row['vs_historical349'];b=row['vs_historical14']
 lines.append(f"| {key} | {a['delta_pp']:+.7f}%p | {a['worst_image_delta_pp']:+.7f}%p | {b['delta_pp']:+.7f}%p | {b['worst_image_delta_pp']:+.7f}%p |")
lines+=['']
lines+=['원본349 대비 약 -0.23%p의 최악 이미지 차이는 기존 lite15에도 있는 경량 경로 차이다. 평균 공식 성능의 근접 유지와 원본349의 모든 이미지별 성능 보존을 동일하게 주장하지 않는다. 이번 신규 후보 판정은 기존 프로토콜의 lite15 기준을 적용했다.','']
lines+=['## 최종 D4 배치 반복 시간','',
'TENT 5-step, EMA 포함 모든 high source, 원본·밝기 보정 두 입력, B4 모든 fold, HRNet, 269, 혼합 계수와 후처리를 보존했다. 8개 D4 view를 순서 그대로 4개씩 묶어 forward한다. 기존 checkpoint FP16 저장과 OpenCV2를 합친 전체 파이프라인을 원래 lite15와 GPU7에서 A-B-A-B-A-B로 비교했다.','']
if batch.get('reference_times'):
 a=batch['reference_times']; b=batch['candidate_times']
 lines+=['| 회차 | 원래 lite15(초) | 15-FP16/D4batch4(초) |','|---|---:|---:|']
 for i,(x,y) in enumerate(zip(a,b),1):lines.append(f'| {i} | {x:.2f} | {y:.2f} |')
 lines += ['',f"- 중앙값: {batch['reference_median']:.2f} → {batch['candidate_median']:.2f}초; 감소 {batch['median_time_reduction_percent']:.2f}%.",f"- 범위: 기준 {min(a):.2f}–{max(a):.2f}초, 후보 {min(b):.2f}–{max(b):.2f}초.",f"- 체크포인트: {batch['reference_checkpoint_bytes']:,} → {batch['checkpoint_bytes']:,} bytes; {100*(1-batch['checkpoint_bytes']/batch['reference_checkpoint_bytes']):.2f}% 감소.",'- 3회 모두 평균·이미지 손실 기준을 통과해야 하며, 중앙값 시간 감소 10% 이상일 때만 탐색 채택으로 판정한다. 공유 서버 부하가 있어 이 결과를 모든 환경의 속도 보장으로 일반화하지 않는다.','']
else:
 lines+=['첫 D4 배치 후보가 품질 기준을 초과해 후속 반복을 중단했다. 이 후보의 속도 개선이나 동등 성능을 주장하지 않는다.','']
if extra:
 lines+=['## 조건부 HRNet FP16 autocast','', '모든 모델·TENT5·raw/style·해상도·계수는 유지하고 HRNet convolution forward를 CUDA FP16 autocast로 수행하고 tile batch를 2에서 8로 늘렸다. 타일 누적 순서는 유지한다. 두 변경을 함께 평가했으며 각각의 속도 효과를 분리하지 않는다. 모델 parameter는 FP32이며 첫 convolution output dtype은 worker 로그로 확인한다. 기본 D4 배치 실험과 구분하여 새 A-B 반복으로 평가했다.','']
 if extra.get('reference_times'):
  lines+=['| 회차 | 원래 lite15(초) | HRAMP 후보(초) |','|---|---:|---:|']
  for i,(x,y) in enumerate(zip(extra['reference_times'],extra['candidate_times']),1): lines.append(f'| {i} | {x:.2f} | {y:.2f} |')
  lines+=['',f"중앙값 {extra['reference_median']:.2f} → {extra['candidate_median']:.2f}초, 감소 {extra['median_time_reduction_percent']:.2f}%.",f"범위: 기준 {min(extra['reference_times']):.2f}–{max(extra['reference_times']):.2f}초, 후보 {min(extra['candidate_times']):.2f}–{max(extra['candidate_times']):.2f}초.",'']
 else: lines+=['품질 기준을 초과해 추가 반복을 중단했다. 상세 변화는 hramp_summary.json과 comparison.json을 참조한다.','']
timing_file=ROOT/'component_timing_summary.json'
if timing_file.exists():
 components=load(timing_file)
 lines+=['## 구성별 벽시계 시간','', '각 값은 3회 중앙값이며 component의 로드·CPU 전처리·GPU forward·확률 저장을 포함한다. GPU kernel 시간으로 부르지 않는다. 그룹 내 첫 실행만 있으면 해당 단회 값이다.','', '| 그룹/구성 | 기준(초) | 후보(초) |','|---|---:|---:|']
 for tag,group in components['groups'].items():
  a=group['references']['medians'];b=group['candidates']['medians']
  for key in a:
   if key in b:lines.append(f'| {tag}/{key} | {a[key]:.2f} | {b[key]:.2f} |')
 lines+=['', '공유 서버 조건은 [component_timing_summary.json](component_timing_summary.json)의 run별 host load·GPU 점유와 원본 telemetry JSONL에 기록했다. 기본 OpenCV 스레드가 많은 기준판의 CPU 대기 증가도 전체 시간에 포함된다. 이 비교에서 HRNet AMP 단독 효과나 환경 독립 속도 보장은 주장하지 않는다.','']
if extra:
 lines+=['HRNet 추가 후보 판정: **'+('탐색 채택 기준 통과' if extra.get('exploratory_adoption_pass') else '미채택')+'**. 평균 손실·최악 이미지 손실·저장 bytes 감소·반복 시간 중앙값 10% 감소를 함께 요구한다.','']
if not accepted:
 lines+=['이번 고정 연산 후보 중 품질과 10% 반복 시간 감소 기준을 함께 충족한 안은 없다. FP16 저장판의 품질 보존과 용량 절감은 확인했지만, 추가 속도 개선을 확정하지 않는다. 기존 15개 기본판을 교체하지 않는다.','']
lines+=['## test10','', '정답은 없다. 아래는 같은 입력 SHA-256·10장 목록을 확인하고 기존 lite15 마스크와 비교한 출력 보존 정도다. test mIoU나 test 성능 유지 점수가 아니다.','', '| 후보 | 전체 픽셀 | 변화 픽셀 | 마스크 일치율 |','|---|---:|---:|---:|']
for label,file in [('15-FP16/D4batch4','batched_test_parity.json'),('15-FP16/D4batch4/HRAMP-batch8','hramp_test_parity.json')]:
 if (ROOT/file).exists():
  test=load(ROOT/file)
  lines.append(f"| {label} | {test['total_pixels']:,} | {test['changed_pixels']:,} | {100*test['agreement']:.8f}% |")
lines+=['']
lines+=['## 추가 의견 검토','',
'10개 계수 0.7~1.0의 평균 80.09~80.12%와 train70 교사 82.9768%(Eutectic Si 65.1677%)는 파일과 일치했다. 하지만 0.8의 최악 이미지 -0.2262%p, 1.0의 -0.6709%p 때문에 평균만으로 추천하지 않는다. 326 Primary Si가 뒤에서 덮어써져도 중간 전체 확률 정규화가 있으므로 완전한 Al3Ni 단독 역할이라고 단정하지 않는다.','',
'증류 학생의 77.5~78.5%는 미측정 예상이다. train70 점수만으로 교사의 암기 여부를 판단할 수 없고, 추가 무라벨 자료가 없어도 GT+soft target 증류는 가능하다. 다만 현 자료는 단일 학생의 80% 유지 근거가 부족하다. B4 fold1 기준 차이는 약 3.12%p, HRNet 단일 출력 기준은 약 1.91%p다. 이번에는 학생 훈련을 시작하지 않았으며 valid/test를 훈련에 사용하지 않았다. [상세 검토](review_notes.md).','',
'## 재현 파일과 보존','',
'- [감사·실행 README](README.md), [기존 기준 보완](protocol.md), [CPU 스레드 프로토콜](cpu_threads_protocol.md), [D4 배치 프로토콜](tta_batch_protocol.md), [HRNet 프로토콜](hramp_protocol.md).','- [공식 채점·이미지별 결과](comparison.json), [반복 시간](batched_summary.json), [FP16 구조·오차 검사](fp16_structural_audit.json), [실제 검증 환경](validation_runtime_versions.json).','- 초기 중단 FP16/TENT3/12를 대체하는 완료 기록은 서버 `/workspace/model349_compress/audit_20261008/runs/{cpu2_fp16_t5,cpu2_fp16_t3,cpu2_fp16_12}/completion.json`이다. 원래 부분 산출물 경로는 보존했다.','- 서버 전체 실험 경로: `/workspace/model349_compress/audit_20261008`. 로컬 결과 아카이브에는 가중치·확률 캐시는 제외하고 코드·설정·마스크·로그·메트릭·해시를 수집한다.','- FP16 구조 검사는 텐서 키/크기, FP32→FP16, finite, 정수/비텐서 보존을 확인하고, 원래 14·15개 가중치의 conversion source SHA도 재확인한다.','']
if accepted:
 lines+=['독립 실행 디렉터리(가중치 포함): `/workspace/model349_compress/audit_20261008/{candidate_package}`. 이는 탐색 채택 후보이며 기존 배포 릴리스를 교체하지 않는다. `bundle_manifest.json`에 코드·설정·가중치의 파일별 SHA-256을 기록했다.','']
candidate_package='model349_lite15_fp16_d4batch4_hramp_candidate' if extra.get('exploratory_adoption_pass') else 'model349_lite15_fp16_d4batch4_candidate'
lines=[line.replace('{candidate_package}',candidate_package) for line in lines]
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print('Wrote',ROOT/'report.md')
