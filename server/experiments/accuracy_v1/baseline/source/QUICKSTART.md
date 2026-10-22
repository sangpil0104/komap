# 학습 실행 안내

현재 `/Data3/sppark/komap`과 conda `komap` 환경은 준비되어 있습니다. 아래 학습 명령부터 실행하면 됩니다.

## 1. 한 번에 학습 → 검증 → 테스트 PNG 생성

```bash
cd /Data3/sppark/komap
conda activate komap
bash scripts/train.sh 8
```

이 명령은 RTX 4090 8장을 사용합니다. GPU 1장만 쓰려면 `bash scripts/train.sh 1`로 실행합니다. 특정 GPU를 선택하려면 앞에 `CUDA_VISIBLE_DEVICES=0` 같은 설정을 붙이세요. 2장·4장도 지원합니다. 동시에 다른 학습이 실행 중이면 사용할 GPU 번호를 지정하세요.

기본 모델은 **CNN/U-Net 기준 모델(E01: gray + 전체 영상 z-score + CDF)** 입니다. 학습 70장, 검증 20장을 사용하고, 검증 이미지별 평균 mIoU가 가장 높은 raw/EMA 가중치를 선택합니다. 모델의 원해상도 경로를 유지하며, 테스트 입력은 학습에 사용하지 않습니다. 원본 데이터는 수정하지 않습니다.

| 단계 | 입력 | optimizer 업데이트 | 신규 모듈 최대 LR |
|---|---|---:|---:|
| s0_patch512 | 중앙 512 + halo 64 = 640 | 2,500 | 3e-4 |
| s1_patch768 | 중앙 768 + halo 64 = 896 | 1,500 | 1e-4 |
| s2_fullframe | 전체 원해상도 | 500 | 1e-5 |

총 **4,500 updates**, effective batch 8입니다. GPU 1장에서는 8회 gradient accumulation, GPU 8장에서는 각 GPU가 1개씩 처리합니다. 1 update를 1 epoch로 표시하지 않습니다. 총 노출량은 36,000개 이미지/패치 표본이며, 70장 기준 약 514회의 이미지당 노출에 해당합니다. 이 예산은 CNN 기준 실험의 시작값이며 최고 성능이 검증된 학습 길이는 아닙니다. 각 단계는 직전 단계의 최고 검증 가중치로 시작하고 optimizer/scheduler를 새로 만듭니다. 자동 조기 종료는 사용하지 않습니다.

## 2. 얼마나 걸리는지 보기

실제 서버의 BF16 순전파·역전파·optimizer update를 단계별로 워밍업 2회 + 측정 5회 수행하고, 전체 검증 20장을 측정했습니다. 벤치마크 가중치는 본 학습에 사용하지 않습니다.

| 사용 GPU | 측정치로 계산한 전체 시간 | 계획용 예상 범위 |
|---|---:|---:|
| RTX 4090 1장 | 약 56분 | 약 45~75분 |
| RTX 4090 8장 | 약 9.5분 | 약 8~15분 |

검증은 250 updates마다 raw/EMA를 모두 평가하며 이 비용도 추정에 포함했습니다. 체크포인트 저장, 초기 로딩, 다른 작업의 GPU 사용, 전체 영상 크기 차이에 따라 달라집니다. 이는 **CNN 기본 설정의 1회 실행** 예상치이며 DINOv3, 여러 seed, 교차검증의 소요 시간은 아닙니다. 짧은 측정에서 얻은 계획용 추정이고 실제 학습 진행률의 ETA가 계속 보정됩니다.

터미널에는 전체 update 진행 막대, 현재 단계, loss, 최고 검증 mIoU, 예상 남은 시간이 표시됩니다.

학습을 시작한 다음 **별도 터미널**에서:

```bash
cd /Data3/sppark/komap
conda activate komap
python -m alphase.cli.monitor --run runs/cnn_unet/official --port 8765
```

브라우저에서 **http://localhost:8765** 를 열면 전체/단계 진행률, 경과·남은 시간, 예상 완료 시각, loss와 mIoU 그래프, 원본/정답/예측 비교가 3초마다 갱신됩니다. VS Code Remote SSH를 사용 중이면 **Ports 탭에서 8765 포트를 전달**하고 표시되는 로컬 주소를 여세요. 모니터를 종료해도 학습은 계속됩니다. 첫 검증은 250 updates 이후입니다.

TensorBoard도 사용할 수 있습니다:

```bash
tensorboard --logdir runs/cnn_unet/official/tensorboard --host 127.0.0.1 --port 6006
```

시간을 다시 측정하려면:

```bash
python -m alphase.cli.benchmark
torchrun --standalone --nproc_per_node=8 -m alphase.cli.benchmark
```

## 3. 저장 위치

| 산출물 | 위치 |
|---|---|
| 전체 단계 중 최고 검증 모델 | `runs/cnn_unet/official/best.pt` |
| 재개용 최신 모델 | `runs/cnn_unet/official/last.pt` |
| 단계별 최고 모델 | `runs/cnn_unet/official/stage_0_best.pt` 등 |
| loss / 검증 기록 | `runs/cnn_unet/official/metrics.jsonl` |
| 실시간 진행 상태 | `runs/cnn_unet/official/status.json` |
| 실제 설정 / 실행 환경 | `runs/cnn_unet/official/resolved_config.yaml`, `environment.json` |
| 독립 실행 검증 결과 | `runs/cnn_unet/official/evaluation/` |
| 테스트 10장 예측 마스크 | `submission/predictions/*_mask.png` |
| 데이터 검사 / ID 목록 / fold | `data/processed/` |
| 시간 측정 근거 | `artifacts/benchmark_cnn_unet_1gpu.json`, `benchmark_cnn_unet_8gpu.json` |

체크포인트는 모델·EMA·optimizer·scheduler·난수 상태·단계·업데이트·학습/검증 ID·팔레트·데이터 hash를 포함합니다. 100 updates마다, 검증 시점과 단계 종료 시 저장합니다. 강제 종료 시 마지막 저장 지점 이후의 업데이트는 다시 수행합니다. 설정, 데이터 또는 GPU 수가 달라진 resume는 거부합니다.

기존 run과 예측 결과를 기본으로 덮어쓰지 않습니다. 다른 실험은 `--run-dir`을 지정하세요. `--overwrite`는 기존 폴더를 별도 백업 이름으로 옮긴 후 새 실행을 만듭니다.

## 4. 학습만 실행하거나 중단 후 이어서 실행

```bash
# GPU 8장, 학습만
torchrun --standalone --nproc_per_node=8 -m alphase.cli.train \
  --config configs/cnn_unet.yaml

# GPU 1장, 학습만
CUDA_VISIBLE_DEVICES=0 python -m alphase.cli.train --config configs/cnn_unet.yaml

# 8장으로 시작한 학습 재개
torchrun --standalone --nproc_per_node=8 -m alphase.cli.train \
  --config configs/cnn_unet.yaml --resume runs/cnn_unet/official/last.pt

# 1장으로 시작한 학습 재개
CUDA_VISIBLE_DEVICES=0 python -m alphase.cli.train \
  --config configs/cnn_unet.yaml --resume runs/cnn_unet/official/last.pt
```

재개 후 학습이 끝나면 다음으로 검증·테스트 출력을 실행합니다. `scripts/train.sh`를 처음부터 사용하면 이 부분도 자동 실행됩니다.

```bash
python -m alphase.cli.validate --checkpoint runs/cnn_unet/official/best.pt
python -m alphase.cli.predict --checkpoint runs/cnn_unet/official/best.pt
python -m alphase.cli.check_submission
```

예측 PNG는 입력과 같은 폭·높이, 지정된 4색 팔레트, alpha=255 RGBA입니다. 기본 파일명은 `*_image.png → *_mask.png`입니다.

## 5. 추가 실험

```bash
# raw gray만 사용하는 대조 실험
python -m alphase.cli.train --config configs/cnn_unet.yaml \
  --run-dir runs/cnn_raw/official --set 'data.image_channels=[gray]'

# 준비한 group fold 중 하나: heldout 이미지 평가 시 --split heldout 사용
python -m alphase.cli.train --config configs/cnn_unet.yaml --split fold0

# 검증을 통해 구조/학습 길이를 정한 후 전체 정답 90장으로 새로 학습
torchrun --standalone --nproc_per_node=8 -m alphase.cli.train \
  --config configs/cnn_unet.yaml --split all90
python -m alphase.cli.predict --checkpoint runs/cnn_unet/all90/best.pt \
  --output submission/final90_predictions

# 검증 시 D4 / 다중 배율 비교: 기본은 모두 끔
python -m alphase.cli.validate --checkpoint runs/cnn_unet/official/best.pt \
  --output runs/cnn_unet/official/evaluation_tta \
  --set inference.d4=true 'inference.scales=[1.0,1.25,1.5]'
```

`all90`은 검증 없이 미리 정한 update 수를 유지하고 마지막 raw 가중치를 `best.pt`에 저장합니다. 이 경우 `best`라는 파일명이 독립 검증 최고점을 의미하지 않습니다. 같은 4,500 updates이면 90장 기준 이미지당 노출량은 400입니다. `valid` 20장을 학습에 포함한 모델로 그 20장을 독립 검증하려 하면 evaluator가 거부합니다.

공식 train/valid 분할에는 파일명 A-prefix 그룹 7개가 겹칩니다. 원본 이미지/픽셀 중복은 없으며, 실제 동일 시편 여부는 확인되지 않았습니다. 따라서 공식 70/20 실험은 유지하고 별도의 A-prefix group 5-fold도 생성했습니다. `fold0`~`fold4` 학습은 fold별 검증 모델 선택을 하는 일반 CV입니다. 보정기용 엄격한 OOF/중첩 검증으로 부르지 않습니다.

## 6. DINOv3와 구현 범위

요청하신 선택에 따라 **공식 가중치가 필요 없는 CNN 기준 학습을 우선 완성**했습니다. DINOv3 v1 결합 모델, 전체 문맥/로컬 ROI 정렬, 동결/해제, L/H+ 설정과 공식 소스도 준비했습니다.

공식 가중치를 받은 뒤 원래 파일명 그대로 두고 다음처럼 경로를 설정할 수 있습니다:

```bash
python -m alphase.cli.train --config configs/hybrid_hplus.yaml \
  --set model.pretrained_checkpoint=/absolute/path/to/original_official_filename.pth
```

가중치가 없으면 명확한 오류로 종료합니다. 공식 소스의 작은 ViT-S를 무작위 초기화해 연결 동작만 검사했으며, **H+/L 공식 사전학습 가중치 로드와 GPU 메모리·학습 성능은 아직 검증하지 않았습니다.** 위 CNN 시간 추정은 DINOv3에 적용할 수 없습니다. 소스 commit은 `artifacts/dinov3_commit.txt`에 고정했습니다.

README의 후속 연구 항목인 ConvNeXt/HRNet, 상호작용 Adapter v2, 앙상블 가중치 탐색, OOF 잔차 보정기, 중첩 CV 자동 실행, 의사라벨링은 이 CNN 실행 구성에 포함하지 않았습니다. 단일 모델의 원해상도 학습·평가·D4/다중 배율·타일 추론·색상 PNG 출력은 구현했습니다. `run_outer_cv`, `generate_oof`, `train_refiner` CLI는 구현된 명령이 아닙니다.

## 7. 환경과 점검

현재 설치: Python 3.11.16, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, TensorBoard. 실제 설치 버전은 `requirements.lock.txt`와 `artifacts/environment.json`에 기록했습니다. PyTorch 조합은 [공식 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/)를 확인했고 DINO 코드는 [공식 저장소](https://github.com/facebookresearch/dinov3)를 사용했습니다.

```bash
# 데이터가 바뀌었을 때만 다시 준비
python -m alphase.cli.prepare --root dataset --out data/processed

# 단위 검사
python -m pytest -q
```

실행 검사 기록은 `artifacts/BUILD_REPORT.md`를 보세요. `runs/smoke/`의 가중치와 예측은 짧은 동작 점검용이며 본 학습 결과나 제출용 모델이 아닙니다. 본 학습용 `runs/cnn_unet/official`은 아직 시작하지 않았습니다.
