# Model349 경량화 재개 감사 (2026-10-08)

이 디렉터리는 기존 릴리스, 가중치, 미완료 산출물을 덮어쓰지 않는 추가 감사 실험이다. 한국어 최종 결과는 `report.md`, 기계 판독 결과는 `comparison.json`, `batched_summary.json`, `hramp_summary.json`, `batched_test_parity.json`, `hramp_test_parity.json`에 기록한다.

## 실행 조건

작업 디렉터리의 GPU 사용자 매뉴얼 PDF 전체를 pdftotext로 읽었다. 공유 GPU/스토리지, /workspace 사용 지침을 따른다. GPU6은 다른 컨테이너 작업이 점유하여 사용하지 않고, 비어 있던 GPU7 P100 한 장만 사용한다. 프로토콜 변경은 결과를 보기 전에 `protocol.md`, `cpu_threads_protocol.md`에 기록했다.

원래 FP16 5-step의 모든 component 20장 추론은 완료되었고 fusion에서 마스크 1장만 저장돼 있었다. 기존 확률과 로그는 보존하고 CPU fusion을 별도 경로로 복구했다. 복구 공식 점수는 80.0655029%이며 full walltime은 알 수 없다. TENT3/12의 과거 완료·부분 산출물은 없었다. 기존 PID 기반 대기 스크립트 대신 flock, 현행 작업 확인, GPU 메모리·점유 재확인을 사용한다.

## 코드

- `infer_audited.py`: 기존 lite14 추론식에 EMA 유지 여부/활성 component 목록을 명시하고, 실제 TENT steps·GPU·fusion 시간·코드 SHA를 completion에 기록한다. B4 제거 시 0 혼합의 누락 소스는 base를 대입해 기존 0 혼합과 동일한 계산을 유지한다.
- `infer_cpu2.py`: 위 코드에 OpenCV 2 threads 제한만 추가한다. torch는 기존 2 threads다. config의 과거 b4_tta/native_tta/hrnet_tta 문자열보다 실제 호출을 기준으로 하며, lite 경로는 B4 tta=False, native none, HRNet tta=False, base 768 D4 raw/style 및 per-image TENT다.
- `prepare_and_run.py`: 기존 source bundle을 읽고 감사용 코드/설정/manifest를 별도 폴더에 만든 뒤 FP16 partial fusion 복구와 기준 실행을 시작한다. GPU7 기준 실행 중 병목 조사 후 이 큐 coordinator만 종료했고, 이미 시작한 기준 추론은 그대로 끝까지 둔다. `queue_transition.json`에 본인 프로세스의 검증된 PID와 이유를 기록했다.
- `run_controlled.py`: 최초 제어 큐다. 사용자 fold 선택 우려를 반영해 진행 중인 추론은 보존하고 `probe_resume.py`로 coordinator만 교체했다. 초기 5개를 마무리하고, TENT3 품질 실패로 12+TENT3 결합은 실행하지 않았다. 축소 fold를 기본 채택하는 반복은 하지 않았다. `run_batched.py`가 15개 모델을 유지한 후보의 A-B 3회 반복과 test parity를 실행한다. 신규 출력 경로가 이미 있으면 중단한다.
- `analyze_runs.py`: 저장된 label PNG와 원본 GT로 IoU를 별도 재계산하여 metrics.json과 1e-12 이내 일치를 요구한다. 입력/교정 SHA, 평균·클래스·이미지별 변화·마스크 일치율을 확인한다. test에는 점수를 계산하지 않는다.
- `capture_audit.py`, `monitor.py`: 프로세스, GPU, 스토리지, 부분 산출물과 5초 간격 GPU7/서버 loadavg 기록.

## 재현

서버 Python: `/workspace/venv-komap/bin/python`, 실험 root: `/workspace/model349_compress/audit_20261008`.
데이터: `/workspace/komap_model1/data/{valid,test}/images`, valid GT: `data/valid/masks`.

새 추론은 각 `bundles/<name>`에서 다음 형식으로 실행한다. 출력은 항상 새 경로를 지정한다.

```bash
/workspace/venv-komap/bin/python -B -u infer_lite.py \
  --input /workspace/komap_model1/data/valid/images \
  --gt /workspace/komap_model1/data/valid/masks \
  --output /workspace/model349_compress/runs/NEW_UNIQUE_PATH \
  --gpus 7 --keep-probabilities
```

단, 시작 전 다른 inference가 없고 GPU7이 비었는지 직접 확인한다. 확인 실패 시 다른 사용자의 작업을 건드리지 않고 중단한다. 분석은 다음과 같다.

```bash
/workspace/venv-komap/bin/python -B analyze_runs.py \
  --root /workspace/model349_compress/audit_20261008 \
  --gt /workspace/komap_model1/data/valid/masks
```

FP16은 저장 정밀도만 낮춘 것이며 모델은 기존 FP32 module에 로드한다. 반복 valid 후보 선택과 같은 이미지의 원본 재추론을 **독립 일반화 검증**이라고 부르지 않는다. test 정답이 없으므로 test mIoU를 주장하지 않는다.

큐 재개 스크립트의 PID/transition은 이번 감사의 진행 중인 프로세스를 식별한 기록이다. 미래 재현에는 이 큐를 그대로 재시작하지 않고, 독립 bundle의 infer_lite.py를 새로운 output 경로로 실행한다. 원래 15개 모델을 보존한 배치 후보의 코드는 infer_batched.py와 style_tta_batched.py이며, 실제 검증 환경과 파일 해시는 결과 JSON에 있다.

조건부 후속 후보는 `infer_batched_hramp.py`와 `hramp_protocol.md`에 정의한다. HRNet CUDA FP16 autocast와 tile batch 2→8을 함께 적용하고 별도 A-B 3회로 비교한다. 저장 FP16과 HRNet 연산 FP16은 서로 다른 변경이다. `component_timing_summary.json`의 시간은 GPU kernel 시간이 아니라 각 component의 입출력·CPU 전처리까지 포함한 walltime이다.

## 이번 실행의 종료 범위

사용자 요청에 따라 test10 출력 확인까지 마치고 추가 GPU 실험을 종료했다. D4 batch4는 A-B 각 3회 중앙값 395.71→395.72초로 시간 개선이 없었다. 조건부 HRNet AMP+batch8은 A-B 각 3회 중앙값 408.67→395.82초(3.14% 감소)로 품질은 통과했지만 사전 10% 시간 기준에는 미달했다. 두 후보 모두 기존 릴리스를 교체하지 않는다. test10 마스크 변화는 각각 462/9,074,964와 537/9,074,964픽셀이며 정답 점수는 없다. 최종 결과와 가중치 무결성 검사는 report.md 및 final_verification.json을 참조한다.
