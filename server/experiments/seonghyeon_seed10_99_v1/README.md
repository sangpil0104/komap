# seonghyeon_v1 — seed 10–99 추가

직전 seed 0–9와 동일한 학습 조건으로 **seed 10–99(90개)**를 추가합니다.
학습은 사용자가 시작합니다. 기존 모델/소스/예측/비교 이미지 200장은 보존합니다.
결과는 **`results/model_results_0926_2/`**에 합쳐 전체 seed 0–99를 비교합니다.

## 실행

기존 전용 Python 환경과 사전학습 가중치가 준비되어 있으므로 설치가 필요 없습니다.
아래 스크립트가 내부에서 `nohup`으로 실행하고 PID와 실행별 로그 경로를 출력합니다.

```bash
bash /Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/start_nohup_6gpu.sh
```

```bash
tail -n 80 -F /Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/outputs/latest.log
```

GPU **0–5**에서 각각 독립 시드 하나씩 최대 6개 병렬 실행합니다.
기존 시드 0–9는 다시 학습하지 않습니다. 총 추가 학습량은 **1,440,000 iterations**입니다.

로그는 모든 시드의 stdout/stderr를 한 곳에 즉시 모으며 `[GPU N seed NN]`으로 구분합니다.
30초마다 `[SUITE]`에 이번 실행 `done=…/90`, 합산 `all_done=…/100`,
진행률·경과 시간·`total_ETA`·`finish_UTC`가 표시됩니다.
재실행에서 완료 시드를 건너뛰면 이번 실행의 분모는 남은 시드 수로 바뀝니다.

직전 10개 시드의 중앙 실행 시간은 **3,646.05초/시드**입니다.
90개를 GPU 6장에 배정한 초기 예상은 **약 15시간 12분**이며 확정 소요 시간이 아닙니다.
큐 대기, 중간 검증, 마지막 D4 평가/비교 이미지 생성까지 포함해 실행 중 실측값으로 갱신합니다.

중지하려면 출력된 PID에 `kill -TERM <PID>`를 실행합니다.
같은 명령을 다시 실행하면 체크포인트·점수·예측·비교 그림이 완성된 시드는 재사용합니다.
미완료 시드는 새 실행 폴더에서 **처음부터 재학습**합니다. 중간 iteration 재개는 없습니다.
미완료 시드의 이전 부분 결과는 `report_history/<시각>/partial_seeds/`에 보존합니다.

## 동일 조건 확인

기준은 실제 학습에 사용된 소스 사본
`seonghyeon_seed10_v1/outputs/20260926T054632_756879Z/source/`입니다.

| 항목 | 고정값 |
|---|---|
| 모델 | U-Net + EfficientNet-B4, ImageNet 인코더 초기화 |
| 입력/샘플링 | gray3, ImageNet 정규화, target_width4000, 면적 비례 샘플링 |
| 학습 | crop1536, batch2, 시드당 16,000 iterations |
| 최적화 | AdamW lr=3e-4, wd=1e-2, warmup300+cosine, CE+Dice, BF16 |
| 데이터 | 같은 train70 / valid20, train만 학습 |
| 검증 | 1,000회마다 plain, 마지막 checkpoint 고정, 최종 D4 TTA |
| workers | 시드당 6개, 같은 RNG/증강 및 DataLoader 설정 |
| 환경 | 기존 `seonghyeon_seed10_v1/.venv/bin/python` 공유 |

`train.py`, `runtime.py`, `vendor/common.py`는 실제 직전 실행 소스와 바이트가 같습니다.
사전 검사에서 소스 SHA256, 학습 인자, 패키지 버전, 데이터 SHA256,
ImageNet 인코더 SHA256이 기존 실행과 일치하는지 확인합니다.
학습 시작점은 모든 시드에서 같은 ImageNet 인코더이고 decoder는 각 시드로 초기화합니다.

이번에 변경한 것은 실행 범위/결과 합산과 비교 그림 생성입니다.
그림은 방금 요청한 **우측 상단 error rate, 제목 28px, 패널 간격 24px**를 유지합니다.
이미지·마스크를 축소하지 않으며 error rate는 `오류 픽셀 수 / 전체 픽셀 수 × 100`입니다.

주 지표와 빈 클래스 정책도 동일합니다.
이미지별 4클래스 평균 → valid20 평균이며 union=0은 IoU=1입니다.
빈 클래스 제외와 0 처리 점수는 참고 열로 남깁니다.

## 결과 합산과 보존

```text
results/model_results_0926_2/
  model_results_0926_2.md       # 0–99 전체 순위, 평균/표본 SD, 시드별 상태
  seed_results.csv             # 기존 10개 + 완료한 추가 시드
  suite.json                   # 전체 100개 상태와 실제 모델 경로
  metrics/seed_00.json ... seed_99.json
  predictions/seed_00/ ... seed_99/
  valid_mask_comparison/seed_00/ ... seed_99/  # 완료 시 총 2,000장
  report_history/<실행시각>/   # 변경 전 md, CSV, suite.json, 재시도 부분 결과

experiments/seonghyeon_seed10_99_v1/outputs/<실행시각>/
  source/                     # 이번 실행 소스 사본
  provenance.json             # 소스/데이터/환경/초기 가중치 기록
  suite_status.json
  seed_10/ ... seed_99/        # last.pt, args.json, valid.json, 로그/상태
```

사용자가 실행할 때 기존 보고서·CSV·suite 상태를 먼저 백업하고 같은 파일을 갱신합니다.
준비 과정에서는 기존 결과를 수정하지 않았습니다. 학습 가중치는 각 실험 실행 폴더에 둡니다.
기존 0–9의 개별 결과 파일을 덮어쓰거나 기존 모델을 교체하지 않습니다.
Markdown의 20장 비교 링크 표는 가독성을 위해 시드를 10개씩 나눠 표시합니다.
중간 실패는 보고서와 종료 코드에 반영됩니다.

읽기 전용 준비 확인:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NO_ALBUMENTATIONS_UPDATE=1 \
  /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python -B -u \
  /Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/run.py --preflight
```

실행 계획만 출력하려면 `--preflight` 대신 `--dry-run`을 사용합니다.
두 옵션 모두 학습을 시작하지 않습니다.
