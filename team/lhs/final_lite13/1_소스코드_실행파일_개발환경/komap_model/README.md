# 고탄성 알루미늄 합금 이미지 상 분할 모델

광학 현미경 이미지에서 Al₃Ni, Primary Si, Al, Eutectic Si 네 상을 픽셀 단위로 나누는 모델입니다.
가중치 13개(R 7개 + E 5개 + F 1개)를 결합하며, 검증용(valid) 20장 이미지별 4상 mIoU는 **80.779378%**입니다.

이 폴더에는 소스 코드, 실행 파일, 개발 환경 파일, 대회 데이터(valid·test)가 들어 있습니다.  모든 경로는 이 폴더 기준 상대 경로입니다.

## 폴더 구조


| 경로                | 내용                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `run_predict.sh`  | 실행 파일: 가중치 13개로 이미지 폴더를 예측해 마스크를 만듭니다                                                                                |
| `run_retrain.sh`  | 실행 파일: 학습용 70장으로 가중치 13개를 처음부터 다시 학습한 뒤 검증용 채점과 평가용 예측까지 합니다                                                         |
| `environment.yml` | 개발 환경 파일 (conda, Python 3.10, PyTorch 2.8 + CUDA 12.8)                                                               |
| `data/`           | 대회 데이터 (valid 20장, test 10장). train 70장은 제출 용량 제한으로 제외                                                                              |
| `inference/`      | 예측 코드: 전체 실행(`predict.py`), R 결합(`r/`), E 예측(`e/`), E 덮어쓰기와 F 보정(`f/`), 설정(`config/`), 가중치 학습 기록(`weight_metadata/`) |
| `training/`       | 학습 코드: R 모델(`r_models/`), E 모델(`e_models/`), F 보정기(`f_refiner/`)                                                     |
| `tools/`          | E3 입력 변환(`make_zcdf_png.py`), 채점(`score_masks.py`)                                                                   |


## 1. 환경

```bash
conda env create -f environment.yml
conda activate komap
```

- 개발·검증 환경: Ubuntu, NVIDIA GPU 16GB 1장, CUDA 12.8 드라이버
- 재학습 첫 실행 때 ImageNet 사전학습 가중치(ResNet34, EfficientNet-B4, HRNet-W48)를 인터넷에서 내려받습니다.

## 2. 예측 (평가용 마스크 만들기)

```bash
bash run_predict.sh WEIGHTS data/test/images out/test 0     # WEIGHTS: 가중치 폴더(R/ E/ F/), 마지막 인자는 GPU 번호
```

- 결과: `out/test/fusion/predicted_masks/`(색 마스크), `out/test/fusion/predicted_labels/`(0~3 라벨)
- 입력 이미지는 대회 데이터와 같은 이름 형식(`*_image.png`)이어야 합니다. 마스크는 `_image`를 `_mask`로 바꾼 이름으로 저장됩니다.
- 결과 폴더(`out/test`)는 새로 만들어집니다. 이미 있으면 덮어쓰지 않고 멈추므로, 다시 실행할 때는 다른 폴더 이름을 주거나 기존 폴더를 지웁니다.
- 검증용 채점:
  ```bash
  bash run_predict.sh WEIGHTS data/valid/images out/valid 0
  python tools/score_masks.py --labels out/valid/fusion/predicted_labels --gt data/valid/masks --output out/valid/metrics.json
  ```
- 마스크 색상(RGB): Al₃Ni (25,76,153), Primary Si (76,178,76), Al (153,127,76), Eutectic Si (204,204,204)
- 제출한 가중치는 SHA256으로 확인합니다(`inference/config/weights_sha256.json`). 다시 학습한 가중치를 쓸 때는 `ALLOW_RETRAINED=1 bash run_predict.sh ...`로 확인을 끕니다.

## 3. 가중치 구성


| 번호     | 모델                                      | 가중치 (`WEIGHTS/…`)                  | 역할                        |
| ------ | --------------------------------------- | ---------------------------------- | ------------------------- |
| R1     | ResNet34 고해상도, 예측 시 이미지별 적응(TENT)       | `R/R1_resnet34_tent/best.pt`       | 시작 확률, Eutectic Si 보조     |
| R2     | ResNet34 원본 해상도                         | `R/R2_resnet34/best.pt`            | Eutectic Si 보조            |
| R3     | EfficientNet-B4 3배 확대                   | `R/R3_b4_x3/fold2/`                | Al₃Ni(R4\~R6과 분담)·Al 보조   |
| R4\~R6 | EfficientNet-B4 가로 4000 (fold 1·3·4 평균) | `R/R4_6_b4_w4000/fold{1,3,4}/`     | Al₃Ni·Primary Si          |
| R7     | HRNet-W48 드문 상 가중                       | `R/R7_hrnet/last.pt`               | Al·Eutectic Si·Primary Si |
| E1\~E2 | EfficientNet-B4 BN 고정 추가 학습 (fold 1·3)  | `E/E1_2_b4/fold{1,3}/`             | E 결합 40%                  |
| E3     | HRNet-W48 세 밝기 입력                       | `E/E3_hrnet_zcdf/`                 | E4와 평균해 E 결합 42%          |
| E4     | HRNet-W48 가로 폭 정규화 (fold 4)             | `E/E4_hrnet/seed1/fold4/last.pt`   | E3와 평균                    |
| E5     | HRNet-W48 가로 폭 정규화 (학습용 70장 전체)         | `E/E5_hrnet_train70/seed5/last.pt` | E 결합 18%                  |
| F      | 3층 CNN 보정기                              | `F/F_refiner/epoch8.pt`            | E가 잘못 바꾼 Al 픽셀을 되돌림       |


처리 순서: R 상별 결합 → 후처리 → E로 Al₃Ni·Al 덮어쓰기 → F 보정.
R의 상별 결합 비율은 `inference/config/r_config.yaml`, E 결합 비율은 `inference/config/predict_config.json`에 있습니다.

## 4. 전체 재학습

재학습에는 학습용 70장이 필요합니다. 대회 train 데이터를 `data/train/images/*_image.png`, `data/train/masks/*_mask.png`에 넣은 뒤 실행합니다.

```bash
bash run_retrain.sh 0            # GPU 1장
bash run_retrain.sh 0,1,2,3      # GPU 여러 장: 독립 학습 묶음을 GPU에 나눠 병렬 실행
```

- 결과는 `runs/retrain/`에 쌓이며, 다시 실행하면 끝난 단계는 건너뜁니다. 로그는 `runs/retrain/logs/`에 학습 묶음(`lane_R1_R2_resnet.log` 등)과 단계(`R3_b4_x3_fold2.log` 등) 이름으로 남습니다.
- 마지막에 가중치 13개를 `runs/retrain/model/weights/`(위 배치)로 모은 뒤, 검증용을 채점하고 평가용 마스크를 만듭니다.
  - 채점 결과: `runs/retrain/pred_valid/metrics.json`
  - 평가용 마스크: `runs/retrain/pred_test/fusion/predicted_masks/`
- 흐름 점검: `SMOKE=1 bash run_retrain.sh 0`은 반복 수를 아주 작게 줄여 끝까지 도는지만 확인합니다. 이때는 학습용 이미지 일부만 넣어도 됩니다(4개 fold에 각각 1장 이상, `DATA=폴더`로 지정).
- 학습 시간: RTX 5080(16GB) 1장 기준 약 14시간(단계별 실측 속도로 환산).


| 번호     | 학습 코드 (`training/…`)                                                              | 학습 데이터            | 시드                       |
| ------ | --------------------------------------------------------------------------------- | ----------------- | ------------------------ |
| R1     | `r_models/resnet/`: ResNet34 기본 학습부터 7단계 이어 학습, 마지막 `pairwise_margin_finetune.py` | 학습용 70장           | 42 (ASPP 768 단계만 123)    |
| R2     | 같은 폴더: R1의 중간 모델에서 6단계 이어 학습, 마지막 `highres_scale_finetune.py`                     | 학습용 70장           | 단계별 2028·239·242·257·269 |
| R3     | `r_models/b4/train_ema.py` (마지막 가중치)                                              | fold 2 학습분        | 3456                     |
| R4\~R6 | `e_models/b4/train.py`                                                            | 각 fold 학습분        | 0                        |
| R7     | `r_models/hrnet/train.py` (기본 HRNet에서 12 epoch 추가)                                | 학습용 70장           | 0, 347                   |
| E1\~E2 | `e_models/b4_bnfreeze/train_bnfreeze.py` (R4\~R6과 같은 fold 모델에서 2,000 step)        | 각 fold 학습분        | 819349                   |
| E3     | `r_models/hrnet/train.py` (기본 HRNet에서 25 epoch, 입력은 `tools/make_zcdf_png.py`)     | 학습용 70장           | 414                      |
| E4     | `e_models/hrnet/train.py`                                                         | fold 4 학습분        | 1                        |
| E5     | 같은 코드                                                                             | 학습용 70장           | 5                        |
| F      | `f_refiner/build_features.py` → `train_refiner.py`                                | E의 학습용 70장 OOF 예측 | 81349                    |


- **fold:** 학습용 70장을 시편(A번호) 단위로 4묶음으로 고정 분할합니다(`training/e_models/config/group_folds.json`).
- **F 학습 입력:** F는 학습에 쓰지 않은 fold 모델의 예측(OOF)으로 학습합니다. 그래서 EfficientNet-B4 가로 4000 fold 1\~4와 HRNet seed 1·2 fold 1\~4를 모두 학습하며, 예측에는 위 13개만 씁니다.

## 5. 재현 범위

- **시드·GPU 연산 편차:** 같은 코드·시드라도 GPU 연산 비결정성 때문에 가중치가 비트 단위로 같아지지 않으며, 구성 모델마다 시드에 따른 성능 편차가 있습니다. 그래서 재학습한 모델의 검증용 점수는 제출 가중치의 점수(80.779378%)와 다를 수 있습니다. 결합 비율과 후처리 설정은 고정이며 재학습 뒤 다시 고르지 않습니다.
- **검증용(valid) 사용 범위:**
  - 결합 비율은 검증용 20장에서 골랐습니다.
  - R1·R2의 ResNet34 이어 학습은 단계마다 검증용 점수로 best.pt를 고릅니다. 학습 자체에는 검증용을 쓰지 않습니다.
  - 나머지 모델은 마지막 가중치를 씁니다.
- **R4\~R6과 E1\~E2:** 같은 fold 모델을 한 번 학습해 R과 E가 함께 씁니다.
- **`inference/weight_metadata/`:** 가중치와 함께 만든 학습 설정 기록(args.json)입니다. 예측 때는 모델 구조 설정만 읽으며, 경로 항목은 `run_retrain.sh`의 결과 폴더 기준으로 적어 두었습니다.

## 6. 이 폴더로 확인한 것

- **제출 가중치로 예측:** `run_predict.sh`로 검증용 20장과 평가용 10장을 원본 이미지부터 예측했습니다(NVIDIA GPU 16GB 1장).
  - 검증용 mIoU: 80.779378%
  - 평가용 마스크 10장: 제출한 `2_test_예측마스크`와 픽셀 단위로 같습니다(9,074,964픽셀 중 차이 0).
- **재학습 흐름:** 학습용 이미지 4장(fold마다 1장)으로 `SMOKE=1 bash run_retrain.sh 0`을 실행했습니다.
  - 가중치 13개 학습 → F 학습 → 가중치 배치 → 검증용 채점 → 평가용 마스크 10장 생성까지 오류 없이 끝났습니다.