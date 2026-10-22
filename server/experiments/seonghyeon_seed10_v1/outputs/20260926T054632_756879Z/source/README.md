# seonghyeon_v1 시드 10개 실험

기존 `experiments/seonghyeon_v1`을 보존하고, 그 폴더의 `run_seeds.sh` 기본값인
**tw4000** 설정을 seed **0–9**로 반복합니다. 이 폴더의 파일만 새로 만들었습니다.
학습 시작은 사용자가 합니다. `--prepare`, `--preflight`, 테스트는 학습하지 않습니다.

## 실행과 로그

이 서버에는 전용 환경과 ImageNet 인코더를 준비했습니다. 다음 명령 하나가
내부적으로 `nohup`을 실행합니다. 터미널을 닫아도 실행이 계속됩니다.

```bash
bash /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/start_nohup_6gpu.sh
```

실행 시 PID와 **그 실행 전용 로그 경로**를 출력합니다. 가장 최근 로그는 아래처럼 봅니다.

```bash
tail -n 80 -F /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/outputs/latest.log
```

직접 `nohup`을 쓰고 싶다면 아래 명령도 같습니다. 위 명령과 둘 다 실행하지 마세요.

```bash
cd /Data3/sppark/komap
mkdir -p experiments/seonghyeon_seed10_v1/outputs/logs
LOG=$(mktemp "$PWD/experiments/seonghyeon_seed10_v1/outputs/logs/seed10_$(date -u +%Y%m%dT%H%M%S)_XXXXXX.log")
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/seonghyeon_seed10_v1/.venv/bin/python -B -u \
  experiments/seonghyeon_seed10_v1/run.py > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
tail -n 80 -F "$LOG"
```

- GPU **0,1,2,3,4,5** 각각에서 독립 모델 하나씩 실행합니다. 6,7번은 사용하지 않습니다.
- 처음 6개를 병렬 실행하고 완료된 GPU부터 나머지 4개를 배정합니다.
- stdout/stderr는 버퍼링 없이 한 로그로 모입니다. `[GPU 2 seed 02]`처럼 구분합니다.
- 학습 100회마다 loss·학습률·진행률·메모리·개별 ETA를 출력합니다.
- 검증은 이미지마다 진행률을 출력합니다. `[STATUS]`와 `[SUITE]`는 30초마다 출력됩니다.
- `[SUITE] total_ETA=HH:MM:SS`는 **대기 시드, 중간 검증, 최종 D4 검증과 그림 저장까지**
  고려한 남은 시간입니다. `finish_UTC`는 예상 완료 시각입니다.
- 시작 직후 ETA는 `measuring`입니다. 첫 iteration부터 추정하고, plain 검증 실측 전에는
  학습 속도로 추정한 임시 검증 비용을 사용합니다. 첫 모델 완료 이후 추정 신뢰도가 올라갑니다.
  단일 GPU 시간을 10개 합산하지 않고, GPU 6개의 남은 작업을 배정해 계산합니다.
- `last_update`는 각 시드 상태가 마지막으로 갱신된 뒤 지난 초입니다. 모델 로딩/검증 도중에는
  iteration 수가 잠시 고정될 수 있으며 `phase`를 함께 보세요.
- `failed=0`, iteration 증가, `[SUITE COMPLETED]`로 진행/완료를 확인합니다.
- 같은 suite의 중복 실행은 잠금으로 막습니다. 재실행은 새 폴더에 처음부터 학습합니다.
  **중간 iteration에서 재개하는 기능은 없습니다.** `last.pt`만 있다고 완료로 간주하지 않습니다.
- 중지: 시작 시 출력된 PID에 `kill -TERM <PID>`. 학습 자식과 데이터 로더도 함께 종료합니다.

전체 로그 외에 각 시드의 `worker.log`, 원자적으로 갱신되는 `status.json`,
전체 `suite_status.json`도 남깁니다. 웹 서버는 사용하지 않습니다.

## 설정과 보존

| 항목 | 값 |
|---|---|
| 모델 | SMP U-Net + EfficientNet-B4, ImageNet 인코더 |
| 입력 | gray3, ImageNet 정규화, 이미지 폭 4000으로 정규화 |
| 학습 | crop1536, batch2, 16,000 iterations/seed, 총 160,000 |
| 옵티마이저 | AdamW lr=3e-4, wd=1e-2, warmup300 + cosine |
| 손실 | CE + Dice, gradient norm clip=1 |
| 정밀도 | BF16 |
| 데이터 | `dataset/train` 70장; `dataset/valid` 20장 |
| 검증 | 1,000회마다 plain; 마지막 체크포인트에 D4 8방향 TTA |
| 선택 | 마지막 모델 고정, valid 점수로 중간 체크포인트를 선택하지 않음 |
| 증강/샘플링 | 원본과 같은 Affine/D4/광도/블러/노이즈, 면적 비례 샘플링 |
| workers | 시드당 6개, OpenCV/BLAS 스레드는 1개로 제한 |

`vendor/`는 원본 파일 사본이며 모델 구조와 추론 함수를 그대로 사용합니다.
새 `train.py`에서 진행 로그, 결과 저장, 시드 처리를 구현했습니다.
원본 worker seed의 `time.time()` 의존성을 없애고 Python/NumPy/PyTorch와
Albumentations의 별도 RNG를 모두 시드에 연결했습니다. 같은 시드는 데이터 로더까지
같은 난수열을 사용합니다. cuDNN deterministic을 켜지만 서로 다른 CUDA/PyTorch/GPU에서
비트 단위 결과 일치까지 보장하는 설정은 아닙니다.

모든 시드에서 **인코더만** 같은 ImageNet 가중치로 초기화하고 decoder는 각 시드로 초기화합니다.
기존 학습 체크포인트를 fine-tuning하는 실험이 아닙니다. 원래 `run_seeds.sh`와 같은 출발점입니다.
학습 전에 CPU에서 ImageNet 인코더를 준비하므로 학습 프로세스에서 다운로드하지 않습니다.

기존 `komap` 환경은 수정하지 않았습니다. `.venv`는 기존 PyTorch **2.7.1+cu128**,
torchvision **0.22.1+cu128**을 공유하고 SMP **0.5.0**, timm **1.0.29**,
Albumentations **2.0.8**, OpenCV-headless **4.13.0.92**를 별도로 설치합니다.
원본 작성 환경의 torch2.11/OpenCV5와는 차이가 있으므로 이전 수치와의 완전 재현 실험은 아닙니다.
이번 10개 시드는 동일한 환경을 사용합니다. 설치 목록은 `environment.freeze.txt`에 기록했습니다.

새 서버/환경에서는 `bash setup.sh`로 준비합니다. `BASE_PY`로 PyTorch가 있는 기반 Python을
지정할 수 있습니다. 준비만 다시 점검하려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NO_ALBUMENTATIONS_UPDATE=1 \
  /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python -B -u \
  /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/run.py --preflight
```

## 결과

실행 시점에 `results/model_results_MMDD_N/`을 새로 할당하며 기존 결과를 덮어쓰지 않습니다.
예를 들어 9월 26일 현재 마지막 번호가 1이면 새 폴더는 `model_results_0926_2`입니다.
실제 경로는 통합 로그와 `outputs/latest_run.json`에 기록합니다.

```text
results/model_results_MMDD_N/
  model_results_MMDD_N.md           # 시드 순위, 평균/표본 SD/최소/최대, 이미지별 링크
  seed_results.csv                  # 모든 점수는 0–1; Markdown에서는 %
  suite.json                       # 성공/실패/진행 상태, 모델 경로
  metrics/seed_00.json ...          # 20장 각각의 점수와 클래스 IoU
  predictions/seed_00/*.png ...     # 원해상도 컬러 마스크
  valid_mask_comparison/seed_00/*.png ... # 시드당 20장, 전체 200장

experiments/seonghyeon_seed10_v1/outputs/<실행시각>/
  source/                          # 실제 실행 소스 사본
  provenance.json                  # 소스/데이터/인코더 SHA256, 환경
  suite_status.json
  seed_00/ ... seed_09/
    job.json, args.json, status.json, worker.log, last.pt, valid.json
```

비교 그림은 원래 해상도를 유지합니다. **좌상 원본 / 우상 오류 위치 / 좌하 예측 / 우하 정답**.
맞은 픽셀은 흰색 `(255,255,255)`, 틀린 픽셀은 해당 위치 정답 색입니다.
Eutectic Si 정답 색은 회색 `(204,204,204)`이고 Primary Si는 초록색입니다.

주 지표는 **이미지별 4클래스 IoU 평균 → valid 20장 평균**입니다.
최근 프로젝트 보고서처럼 union=0이면 IoU=1, 정답이 없어도 오검출하면 IoU=0입니다.
과거 seonghyeon 지표와 비교할 수 있게 빈 클래스 제외 점수와 union=0을 0으로 두는 점수도
함께 기록합니다. Global mIoU로 정렬하지 않습니다.

완료한 시드부터 보고서가 갱신되며 실패가 있으면 최종 상태를 `failed`로 명시합니다.
체크포인트·20개 점수·20개 예측·20개 비교 그림이 모두 있어야 성공으로 셉니다.
실제 학습이 끝나기 전에는 결과 수치와 모델 파일이 생성되지 않습니다.
