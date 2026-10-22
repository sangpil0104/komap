# Seed 32 구조 기반 경계 ablation — 8 GPU 순차 실행

**각 모델을 새로 학습합니다. seed 32의 학습된 분할 체크포인트에서 이어 학습하지 않습니다.**
기준 구조는 `EfficientNet-B4 + U-Net`, 인코더 초기화는 기존 실험과 같은 로컬 ImageNet 가중치,
디코더 및 새 모듈은 난수 시드 32로 초기화합니다. 모든 T 실험이 독립적으로 출발합니다.
`contrast`는 비교할 실험 이름이며 그 실험의 체크포인트를 가져온다는 뜻이 아닙니다.

사용자가 아래 명령으로 학습을 시작합니다. 준비·검증 과정에서는 실제 데이터로 학습하지 않습니다.

```bash
bash /Data3/sppark/komap/experiments/seed32_boundary_ablation_v1/run.sh
```

한 실험이 **GPU 0–7 전체에서 DDP 학습 → 마지막 가중치 저장 → valid20 평가·그림 생성**을 끝내면
다음 실험을 시작합니다. 실험 여러 개를 동시에 실행하지 않습니다. 실행 환경은 항상 conda `komap`입니다.
터미널의 두 tqdm 막대에 suite 진행률과 현재 실험 진행률, 경과 시간, 남은 시간이 표시됩니다.
학습 중 전체 잔여 학습 시간은 실측 update 속도로 추정합니다. 최종 평가·저장 시간은 별도이며
전체 suite 막대의 ETA는 완료한 실험들의 실측 시간을 사용합니다. 웹 서버는 실행하지 않습니다.

계획 또는 환경만 확인하는 안전한 명령:

```bash
bash experiments/seed32_boundary_ablation_v1/run.sh --dry-run
bash experiments/seed32_boundary_ablation_v1/run.sh --preflight
```

## 실험 설계

학습 모델 10개, 기존 모델의 추론 비교 3개입니다. 각 변경 실험과 명시된 대조군은 **설정 항목 한 개만** 다릅니다.
이 조건을 실행 전 코드로 검사합니다. 결과를 보며 조합을 자동 선택하거나 다음 모델의 초기화로 사용하지 않습니다.

| ID | 비교 대상 | 변경하는 한 요소 |
|---|---|---|
| T00_control | 새 공통 기준 | 기존 B4 U-Net과 증강으로 신규 학습 |
| T01_native_geometry | T00 | 원본·확대 해상도의 좌표가 일치하는 crop 전처리 |
| T02_native_loss | T01 | 원본 크기로 복원한 예측에 `0.5 × (NLL + Dice)` 추가 |
| T03_boundary_sampling | T01 | crop 중심의 35%를 train 정답의 상 경계에서 추출 |
| T04_boundary_weight | T01 | 경계 띠의 CE 가중치만 1에서 2로 변경 |
| T05_detail_branch | T00 | 축소 없는 모델 입력 해상도 특징 경로 추가 |
| T06_no_gaussian | T00 | Gaussian blur만 제거; 같은 noise·나머지 증강 유지 |
| T07_native_refiner | T02 | 원본 해상도의 영상+클래스 확률 보정 모듈 추가 |
| T08_refiner_no_gray | T07 | 보정 모듈의 영상 입력만 0으로 차단; 구조·파라미터 수 동일 |
| T09_bilinear_decoder | T00 | 디코더 업샘플링만 nearest에서 bilinear로 변경 |

T01을 별도로 둔 이유: 원본 라벨에 추가 손실을 적용하려면 확대 crop과 원본 crop의 좌표 대응이 필요합니다.
이 전처리 자체의 효과를 T01−T00으로 분리하고, T02/T03/T04는 T01과 비교합니다.
T07의 보정 모듈에는 원본 크기의 감독 신호가 필요하므로 같은 손실을 쓰는 T02와 비교합니다.
T08은 영상의 실제 세부 정보가 보정에 기여하는지 확인합니다.

| ID | 비교 대상 | 추론 조건 변경 — 학습 없음 |
|---|---|---|
| R00_seed32 | 보존된 최고 모델 | 기존 seed32 `last.pt`, width4000, D4, area 복원 |
| R01_no_tta | R00 | D4만 제거 |
| R02_bilinear_restore | R00 | 마지막 원본 해상도 복원만 area→bilinear |

R 실험에서만 기존 seed32의 분할 가중치를 읽습니다. T 실험의 초기화 경로와 분리되어 있습니다.
실행 순서는 R00, R01, R02, T00 … T09입니다. 각 R 실험도 GPU 8개에 valid20을 나눠 평가합니다.

## 공통 학습 조건과 원본 실험과의 차이

| 항목 | 새 실험의 공통값 |
|---|---|
| 학습/검증 | 공식 train70 / valid20; valid 라벨을 학습·샘플링에 사용하지 않음 |
| 인코더 | EfficientNet-B4, 기존과 같은 ImageNet encoder 파일 |
| 초기화 | seed=32, 모든 T 실험에서 새 디코더; fine-tuning 아님 |
| 입력 | gray3, `(gray / 255 - 0.449) / 0.226` |
| 기준 확대 / crop | target_width=4000 / 1536×1536 |
| GPU / 배치 | DDP 8 GPU, GPU당 1 crop, 유효 배치 8 |
| BatchNorm | 기존 B4/U-Net의 BN을 SyncBatchNorm으로 변환; 8 GPU 통계 공유 |
| 학습 길이 | 16,000 optimizer updates = 128,000 crops/모델 |
| 손실 | CE + Dice; Dice의 교집합·분모는 GPU 8개의 통계를 미분 가능한 합으로 계산 |
| 최적화 | AdamW, 전체 학습 파라미터 lr=3e-4, weight_decay=0.01 |
| 스케줄 | warmup 300 updates + cosine; gradient norm clip=1 |
| 정밀도 | BF16 autocast; CE/Dice/softmax는 FP32 |
| 모델 선택 | 고정 마지막 update의 `last.pt`; valid 기반 조기종료·best 선택 없음 |
| 최종 추론 | 전체 영상 width4000, pad32 reflect, D4 8방향 확률 평균, area로 원본 복원 |

**원본 seed32는 배치 2, 16,000 updates였습니다.** 한 실험에 GPU 8개를 실제로 쓰기 위해 새 공통 배치는 8입니다.
따라서 새 모델은 원본의 4배인 128,000 crop을 봅니다. R00과 T00의 차이를 구조 변경의 효과로 해석하면 안 됩니다.
구조·손실·샘플링의 효과는 위 표의 T 대조군끼리 비교합니다. 배치 증가에 맞춰 학습률을 자동으로 키우지는 않습니다.
T 모델은 모두 동일한 update 수·샘플 순서·초기화 규칙을 공유합니다.

원본과 같은 총 32,000 crop 노출 예산을 별도 프로토콜로 쓰려면 시작할 때 `--steps 4000`을 지정할 수 있습니다.
이 경우 모든 T 실험이 같은 4,000 updates를 사용하지만, optimizer update 수가 원본보다 줄어듭니다.
길이가 다른 suite의 점수는 단일 요소 ablation으로 비교하지 않습니다.

## 좌표와 경계 학습

T00 계열은 보존된 소스와 같은 연산 순서를 사용합니다: 이미지 면적 비례 샘플링, width4000 배율,
Affine(scale0.7–1.4, rotate±180°, p0.7), RandomCrop, D4, 밝기·대비, 감마, blur/noise.
새 실행의 샘플 RNG는 `(seed, sample_index, stream)`으로 고정하므로 worker 스케줄이나 재개에 좌우되지 않습니다.
원본 학습의 worker RNG를 비트 단위로 재현한다는 뜻은 아닙니다.

T01 계열의 `aligned_native_roi_v1`은 먼저 원본 좌표에서
`round(1536 × original_width / 4000)` 크기의 영역을 고르고, 동일한 affine 좌표로 원본 영상과 라벨을 만듭니다.
그 뒤 공통 D4, 밝기·대비·감마를 적용하고 영상을 bilinear로 확대합니다. 확대 라벨은 nearest-exact를 사용합니다.
blur/noise는 확대된 detail 입력에 적용합니다. 원본 보정 분기는 그 전의 원본 해상도 영상을 받습니다.
영상 밖은 반사 패딩, 라벨의 무효 영역은 255이며 손실에서 제외합니다.
이 전처리 순서 차이는 T01이라는 별도 대조군에서 측정합니다.

경계 띠는 서로 다른 유효 클래스가 이웃하는 픽셀을 표시한 뒤 원본 좌표에서 2픽셀 팽창합니다.
사진 바깥 테두리나 ignore 영역은 상 경계로 사용하지 않습니다.
T04의 가중치는 네 클래스 모두에 동일하게 적용하고, 가중치 합으로 정규화한 CE를 씁니다. Dice는 그대로입니다.
T02의 native 손실은 학습 예측 확률을 `area`로 원본 ROI 크기에 맞춰 원본 라벨과 비교합니다.
원본 ROI마다 NLL+Dice를 구하고 GPU 평균합니다. valid 라벨을 보정기에 전달하는 경로는 없습니다.

## 추가 모듈

T05: 정규화된 입력의 한 채널 → 해상도를 유지하는 `3×3 Conv + GroupNorm + SiLU` 두 번(16채널).
이를 U-Net의 마지막 16채널 특징과 연결하고 16채널 합성곱 후 4채널 보정 logit을 만듭니다.
마지막 층은 0으로 초기화하며, 기존 segmentation logit에 더합니다. 여기서 입력 해상도는 확대된 모델 입력 크기입니다.

T07: 원본 사진 해상도의 gray 1채널 + 복원된 클래스 확률 4채널 → 24 → 24 → 16채널 CNN → 4채널 보정량.
모든 합성곱은 stride1이며 마지막 층은 0으로 초기화합니다.
`softmax(log(P_base + eps) + delta)`로 최종 확률을 만듭니다. 실제 코드는 log 전에 최소값을 clamp합니다.
학습은 원본 ROI에서, 추론은 D4를 평균하고 원본 크기로 복원한 확률과 원본 사진을 함께 사용합니다.
전체 모델을 함께 학습하므로 학습된 예측 캐시를 별도로 사용하는 2단계 refiner가 아닙니다.
T08은 동일 모듈에서 gray 입력만 0으로 만들어 확률만으로 보정하는 대조 실험입니다.

## 결과와 재개

실행마다 새 suite 폴더를 만듭니다. 소스 사본, 환경, 데이터·초기 가중치 SHA-256을 저장합니다.
1,000 updates마다 optimizer/scheduler 및 rank별 RNG를 포함하는 체크포인트를 **새 파일**로 저장합니다.

```text
experiments/seed32_boundary_ablation_v1/outputs/<UTC시각>/
  source/                     # 실행에 실제 사용하는 고정 소스
  plan.json                   # 모든 모델과 비교 대상, 초기화, 데이터/환경 해시
  logs/<실험>_<시각>.log
  trials/<실험>/
    checkpoints/step_001000.pt ... step_016000.pt
    last.pt                   # 새로 학습한 최종 모델
    trained.json
    training.jsonl
    progress.json
    complete.json
  comparison.md / comparison.csv
```

각 실험의 평가가 끝나면 `results/model_results_MMDD_번호/`에 다음을 저장합니다.

```text
model_results_MMDD_번호.md
summary.json / per_image.csv / provenance.json / complete.json
predictions/<valid_image_id>.png
per_image/<valid_image_id>.json
valid_mask_comparison/<valid_image_id>.png
```

비교 그림은 원본 해상도를 유지한 2×2 배치입니다: 좌상 원본, 우상 오류 위치, 좌하 예측, 우하 정답.
우상은 맞은 픽셀을 흰색, 틀린 픽셀을 그 위치의 정답 색으로 표시합니다.
상단에 image mIoU와 `100% − image mIoU`를 표시합니다. 후자는 픽셀 오류율이 아닙니다.
유효 검증 ID 20개와 결과 파일 해시를 검사한 뒤 완료 상태를 기록합니다.

주 지표는 이미지별 4클래스 IoU 평균을 valid20에서 평균하며 **union=0인 클래스 IoU는 0**입니다.
Global mIoU는 계산하지 않습니다. 원본 기준 1·2·3픽셀 경계 띠의 오류율은 원인 분석용으로만 남깁니다.

중단되면 터미널에 출력된 경로로 재개합니다.

```bash
bash experiments/seed32_boundary_ablation_v1/run.sh --resume /Data3/sppark/komap/experiments/seed32_boundary_ablation_v1/outputs/<UTC시각>
```

완료 실험은 결과와 가중치 해시를 확인해 건너뜁니다. 미완료 학습은 **그 새 실험 자체의** 마지막 저장 update에서 재개합니다.
저장 전의 최대 999 updates는 다시 계산할 수 있습니다. 원본 seed32 가중치에서 미세조정하는 것과 다릅니다.
평가만 중단됐다면 최종 가중치에서 평가를 다시 수행하며, 이전 부분 결과는 보존하고 새 results 폴더를 만듭니다.
OS 파일 잠금으로 같은 suite의 중복 실행을 막고, 실패한 실험은 종료 코드를 전파하며 다음 실험을 시작하지 않습니다.
Ctrl+C, SIGTERM, SIGHUP으로 중단하면 실행기가 자신이 시작한 torchrun 프로세스 그룹을 종료합니다.

일부만 실행할 때는 필요한 대조군도 자동 포함됩니다.

```bash
bash experiments/seed32_boundary_ablation_v1/run.sh --experiments T07_native_refiner
```

기존 모델·소스·결과의 보존 여부는 `preservation.json`의 SHA-256으로 시작/완료 시 검사합니다.
검증 결과와 실제 실행하지 않은 범위는 [BUILD_REPORT.md](BUILD_REPORT.md)를 참고하세요.
