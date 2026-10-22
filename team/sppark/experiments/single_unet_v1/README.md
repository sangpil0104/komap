# single_unet_v1 — 기본 단일 U-Net 해상도 실험

기존 모델·소스·결과를 수정하지 않는 독립 실험이다. **학습은 사용자가 아래 `train` 또는 `suite` 명령을 실행할 때만 시작한다.**

## 모델과 비교 조건

- torchvision EfficientNet-B4, ImageNet `IMAGENET1K_V1` encoder + 기본 U-Net decoder.
- encoder endpoint: stride 2/4/8/16/32, channel 24/32/56/160/448.
- decoder channel: 256/128/64/32/16. bilinear upsample → skip concat → Conv-BN-ReLU 두 번.
- 마지막 1×1 convolution이 4-class logit을 출력한다. 별도의 boundary head/ASPP/attention/앙상블은 없다.
- 흑백을 3채널로 복제하고 ImageNet mean/std로 정규화한다. 분류용 center crop/resize는 사용하지 않는다.
- 클래스 순서: Primary Si, Al3Ni, Eutectic Si, Al. 기존 데이터의 정확한 4색 RGBA 팔레트를 유지한다.
- 작은 micro-batch와 activation checkpoint 재계산을 위해 encoder BN running statistics는 고정한다. affine parameter는 학습한다. 모든 비교 실험에서 동일하다.
- 이는 torchvision backbone에 구현한 기본 U-Net이다. 동료의 `smp` 구현/가중치/증강 설정을 그대로 재현한 모델은 아니다.

| 실험 | 배율 | 원본 중심 시야 | 중심 입력 | halo 포함 실제 입력 | 손실 |
|---|---:|---:|---:|---:|---|
| S0 | 1 | 512×512 | 512×512 | 640×640 | CE + Dice |
| S1 | 2 | 512×512 | 1024×1024 | 1280×1280 | CE + Dice |
| S2 | 3 | 512×512 | 1536×1536 | 1920×1920 | CE + Dice |
| S3 | 사용자 지정 1/2/3 | 512×512 | 선택 배율 | 선택 배율 | CE 경계 가중치 +0.2, Dice 동일 |

모든 실험은 같은 seed의 같은 ImageNet 초기 가중치/decoder 초기화에서 독립적으로 시작한다. S3도 기존 run을 이어 학습하지 않는다. 원본 이미지 선택·crop 위치·증강 난수는 같은 seed/index에서 동일하다. 원본 시야를 고정하고 배율만 바꾼다. S3의 GT 경계 띠는 **확대 전 원본 2px**로 생성한다.

기본 일정: 8,000 optimizer updates, effective batch 8, micro-batch 1, AdamW(encoder 1e-4 / decoder 3e-4, WD 0.01), warmup 250 + cosine, BF16, gradient clipping 1.0. D4 학습 증강과 약한 밝기/대비/감마만 적용한다. 이미지 균등 샘플링, halo 부분과 영상 밖 라벨은 ignore=255이다. Dice는 각 이미지에서 GT가 있는 클래스 평균이며 CE가 없는 클래스의 오검출도 억제한다. 픽셀 CE와 Dice 모두 이미지별 계산 후 배치 평균한다.

**고정 예산 마지막 checkpoint를 평가**한다. validation으로 best/EMA/raw를 선택하지 않는다. 학습 중에는 training.jsonl에 CE/Dice/LR/경과 시간/ETA를 기록한다. 원본 화질을 유지한 입력 확대이며 초해상도 모델을 쓰는 것은 아니다.

## 평가 기준

주 지표는 이미지별 4클래스 IoU 평균을 구한 뒤 validation 20장 평균한 값이다. **정답·예측 모두 없는 클래스의 IoU는 1**로 처리한다. 사용자가 판단을 맡긴 뒤 선택한 기본 규칙이며 대회 공식 코드가 확인되었다는 뜻은 아니다.

- `image_miou`, `miou_one`: 기본 주 지표(항상 4클래스 평균).
- `miou_zero`: 빈 클래스 IoU=0, 항상 4클래스 평균.
- `miou_exclude`: 빈 클래스 제외, 기존 보고서와 비교하기 위한 값.
- `--empty-union`으로 정책을 바꿀 수 있으나 비교하는 run끼리는 같아야 한다. 기존 78.x% exclude 결과와 기본 주 지표를 직접 비교하면 안 된다.
- Global mIoU는 계산/선택 기준으로 사용하지 않는다.
- 기본 추론은 **단일 모델·단일 배율·TTA 없음**, native core512 / halo64 / 50% overlap / Gaussian 결합의 타일 추론이다. 배율이 커도 원본 문맥 크기는 고정된다.
- 클래스 확률을 bilinear로 원본 크기에 복원한 뒤 argmax한다. 라벨 학습 확대에는 nearest 보간만 사용한다.
- 1/2/4px 경계 오류율, Boundary IoU, 외곽/내부 16/32/64px image mIoU와 오류율, 클래스별 precision/recall 및 혼동 행렬을 함께 저장한다. 모든 폭은 원본 픽셀 기준이다.
- Boundary IoU 보조 지표는 클래스별 내부 경계 띠를 사용하고 양쪽 빈 클래스는 제외한다. 주 image mIoU와 집계 정책이 다르므로 별도 이름으로 기록한다.

## 준비와 실행

현재 `komap` 환경(torch 2.7.1, torchvision 0.22.1)을 사용한다. 추가 패키지 설치가 필요하지 않다. Bash wrapper의 Python 경로는 `KOMAP_PYTHON`으로 바꿀 수 있다.

```bash
cd /Data3/sppark/komap
conda activate komap

# 학습 없는 준비/검사
python -B experiments/single_unet_v1/run.py prepare
python -B experiments/single_unet_v1/run.py preflight

# 사용자 실행: S0/S1/S2를 순차 학습하고 각각 평가·결과 폴더 생성
CUDA_VISIBLE_DEVICES=0 bash experiments/single_unet_v1/run_suite.sh --gpus 1

# 또는 선택한 4 GPU에서 각 모델을 DDP 학습, effective batch는 계속 8
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/single_unet_v1/run_suite.sh --gpus 4

# 개별 모델
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py train S1 --seed 2026

# 3개 seed를 독립 평가 (앙상블하지 않음)
CUDA_VISIBLE_DEVICES=0 bash experiments/single_unet_v1/run_suite.sh --seeds 2026 2027 2028

# 선택적 S3: 2배 설정을 골랐다면 동일 예산으로 새로 학습
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py train S3 --scale 2 --seed 2026
```

GPU 개수는 1/2/4/8을 지원한다. `CUDA_VISIBLE_DEVICES`로 사용할 GPU만 지정한다. 진행률·경과 시간·ETA는 터미널 tqdm으로 표시된다. 웹 서버/대시보드는 사용하지 않는다.

매번 timestamp가 다른 새 run을 만든다. 명령 재실행은 자동 재개가 아니라 새 실험이다. 실행 중 출력되는 `Run: ...` 경로를 보관한다. `--updates`를 바꾸면 비교 실험 모두에 같은 예산을 적용해야 한다.

## 중단 재개와 평가

```bash
# 실제 Run 경로로 치환
run_dir=/Data3/sppark/komap/experiments/single_unet_v1/outputs/runs/S1_x2_seed2026_official_실행시각

# 원래와 같은 GPU 개수로 최근 저장 update부터 재개
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py train --resume "$run_dir" --gpus 1

# 동일 checkpoint의 외곽 처리 5조건 비교 후 새 results 폴더 생성
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py compare-borders "$run_dir"

# 전체 영상 추론과 D4는 별도 비교
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py evaluate "$run_dir" --mode full --pad 64
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py evaluate "$run_dir" --d4

# 최종 선택 모델로 test 10장 마스크 생성 (test 점수 없음)
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py predict "$run_dir"

# A-ID를 분리한 기존 group fold에서 독립 초기화 학습
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py train S1 --split fold0
```

외곽 비교는 full-frame 기본 padding, full-frame 사방 reflect64, tiled halo64, tiled halo128, tiled halo64+격자 offset128의 5조건이다. 단순 타일 격자 이동은 별도의 TTA/평균이 아니라 독립 추론 비교다. 기본 full-frame도 오른쪽/아래 32배수 padding은 유지한다. full-frame 3배에서 메모리가 부족하면 명확히 실패하며 타일로 몰래 바꾸지 않는다. 개별 `evaluate` 명령으로 메모리에 맞는 조건부터 확인할 수 있다.

재개 시 모델·optimizer·rank별 RNG·다음 sample index를 복원한다. 이전 checkpoint 파일은 교체하지 않는다. 마지막 checkpoint 이후 저장되지 않은 update는 다시 수행한다. training.jsonl은 재개 전 로그를 보존하므로 rollback된 update가 중복될 수 있다. 완료된 run은 다시 학습하지 않으며 재개 명령에서는 평가만 재실행한다.

## 저장 위치

```text
experiments/single_unet_v1/
  configs/base.yaml
  weights/                         # 공식 ImageNet 가중치, 기존 weights 경로 미사용
  outputs/verification/            # 학습 없는 사전 점검
  outputs/runs/<새 실행 이름>/
    config.json, dataset.json
    source/, source_hashes.json    # 실행 소스 snapshot
    training.jsonl
    checkpoints/update_XXXXXX.pt  # optimizer/RNG 포함, 1000 updates마다
    latest.json, completed.json
    evaluations/<새 평가 이름>/
      request.json, summary.json, per_image.csv
      masks/, comparisons/
results/model_results_MMDD_N/
  model_results_MMDD_N.md
  summary.csv, provenance.json
  <각 평가>/masks/, comparisons/, per_image.csv, summary.json
```

학습 완료 후 검증·리포트 생성이 자동으로 이어진다. 검증만 실패했다면 다시 학습할 필요 없이 `evaluate RUN`을 실행하면 된다. 각 완료 실험 보고서와 suite 전체 비교 보고서는 별도의 새 번호를 할당한다. 기존 결과 번호·파일은 덮어쓰지 않는다. 보존된 PhaseCNN Plain 점수는 별도 참고값이며 입력 채널/추론 조건 차이를 명시한다.

소스를 바꾸면 기존 run 평가/재개는 hash 검사에서 중단된다. 해당 run의 보존 소스를 사용할 때는 `python RUN/source/run.py evaluate RUN`처럼 실행한다. resume에는 원래 GPU 개수와 split/seed/예산을 유지한다.

## 학습 없는 검증

```bash
python -B -m pytest -q experiments/single_unet_v1/tests
CUDA_VISIBLE_DEVICES=0 python -B experiments/single_unet_v1/run.py preflight --profile
```

`--profile`은 무작위 합성 입력으로 S0/S1/S2 순전파·역전파와 CUDA 최고 메모리를 검사한다. optimizer 생성/step과 실제 데이터 학습은 하지 않는다. 실제 학습 메모리는 optimizer 상태와 DDP 버퍼 때문에 검사값보다 증가한다.

구조 참고: [Torchvision EfficientNet-B4](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.efficientnet_b4.html).
경계 지표 참고: [Boundary IoU, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html).
