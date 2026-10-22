# lite13 확률 재조정 재현 패키지 (고탄성 알루미늄 합금 이미지 상 분할)

lite13 확률 재조정 모델(가중치 13개: R 7개 + E 5개 + F 1개, 검증용 20장 mIoU 80.779378%)의
**예측 코드, 전체 재학습 코드, 개발 환경 파일**입니다. 모든 경로는 이 폴더 기준 상대 경로이며,
`data/` 아래 train·valid·test만 있으면 서버 경로 없이 재학습과 예측이 됩니다.

lite13과 가중치 13개는 같고, 모델 확률을 섞는 비율만 다릅니다.

| 설정 (파일) | 항목 | lite13 | 재조정 |
|---|---|---|---|
| R 결합 (`inference/selected_R_lite13.yaml`) | `al_gamma` | 0.6 | 1.0 |
| | `al3ni_gamma` | 0.7 | 1.0 |
| | `eut_edge_blend` | 0.25 | 0.125 |
| E 결합 (`inference/selected_config.json`) | `family_weights` (B4 / HR 묶음1 / HR 묶음2) | 0.5 / 0.25 / 0.25 | 0.4 / 0.42 / 0.18 |

- 재조정 비율은 검증용 20장에서 골랐습니다. 그래서 검증용 점수 상승(+0.058%p)은 일반화 성능 향상의 근거가 아닙니다.
- `inference/infer_expert.py`는 B4 비율 0.5만 허용하던 확인을 "비율 합 1"로 넓혔습니다. 계산식은 같습니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `run_predict.sh` | 가중치 13개로 이미지 폴더 예측(마스크 생성) |
| `run_retrain.sh` | 학습용 70장으로 13개 가중치 전체 재학습 → 검증용 채점 → 평가용 예측 |
| `environment.yml` | 개발·검증에 쓴 conda 환경(Python 3.10, PyTorch 2.8 + CUDA 12.8) |
| `data/` | 대회 데이터를 넣는 곳(`data/README.md` 참고) |
| `inference/` | R 결합·후처리(`parent_package/Model349_exh14`), E 예측(`infer_expert.py`), F 보정(`parent_package/infer_fusion.py`), 설정(`selected_*.json/yaml`) |
| `training/` | 모델별 학습 코드(아래 표) |
| `tools/` | N3 입력 변환(`make_zcdf_png.py`), 채점(`score_masks.py`) |

## 1. 환경

```bash
conda env create -f environment.yml
conda activate komap
```

- 검증 환경: Ubuntu, NVIDIA RTX 5080 16GB 1장, 드라이버 CUDA 12.8.
- 학습 첫 실행 때 ImageNet 사전학습 가중치(ResNet34, EfficientNet-B4, HRNet-W48)를 인터넷에서 내려받습니다.

## 2. 데이터

```
data/train/images/*_image.png (70)   data/train/masks/*_mask.png
data/valid/images/*_image.png (20)   data/valid/masks/*_mask.png
data/test/images/*_image.png  (10)
```

## 3. 예측 (제출 마스크 만들기)

```bash
bash run_predict.sh WEIGHTS data/test/images out/test 0
```

- 결과: `out/test/fusion/predicted_masks/`(색 마스크), `out/test/fusion/predicted_labels/`(0~3 라벨).
- 검증용 채점: `python tools/score_masks.py --labels out/valid/fusion/predicted_labels --gt data/valid/masks --output out/valid/metrics.json`
- `WEIGHTS` 폴더 배치(공개 가중치 13개, SHA256은 `inference/selected_assets.json`):

```
WEIGHTS/R/restart61_pairwise_margin/best.pt                              R1
WEIGHTS/R/model269_fullstats_1x_extended/best.pt                         R2
WEIGHTS/R/model326_hybrid_fullstats_group_oof/fold2/{last.pt,args.json}  R3
WEIGHTS/R/model329_fullstats_width4000_uniform_group_oof/fold{1,3,4}/{last.pt,args.json}  R4~R6
WEIGHTS/R/model347_hrnet_inverse_ce_finetune/last.pt                     R7
WEIGHTS/E/model329_ws/fold{1,3}/{last.pt,args.json}                      E1~E2
WEIGHTS/E/model414_seed414/{last.pt,args.json}                           E3
WEIGHTS/E/model392/seed1/fold4/last.pt                                   E4
WEIGHTS/E/model392_train70/seed5/last.pt                                 E5
WEIGHTS/E/refiner/epoch8.pt                                              F1
```

재학습한 가중치를 쓸 때는 SHA256 확인을 끕니다: `ALLOW_RETRAINED=1 bash run_predict.sh ...`
(`run_retrain.sh`가 마지막에 자동으로 실행합니다).

## 4. 전체 재학습

```bash
bash run_retrain.sh 0            # GPU 1장
bash run_retrain.sh 0,1,2,3      # GPU 여러 장: 독립 학습 레인을 GPU에 나눠 병렬 실행
```

- 결과는 `runs/retrain/`에 쌓이고, 다시 실행하면 끝난 단계는 건너뜁니다.
- 마지막에 13개 가중치를 `runs/retrain/model/weights/`(위 배치)로 모은 뒤 검증용을 채점하고 평가용 마스크를 만듭니다.
  - 채점 결과: `runs/retrain/pred_valid/metrics.json`
  - 평가용 마스크: `runs/retrain/pred_test/fusion/predicted_masks/`
- `SMOKE=1 bash run_retrain.sh 0`: 반복 수를 아주 작게 줄여 전체 흐름만 확인합니다(성능 없음).
- 학습량: 기존 기록 기준 P100 약 80 GPU시간입니다.

| 번호 | 모델 | 학습 코드 (`training/…`) | 학습 데이터 | 시드 |
|---|---|---|---|---|
| R1 | ResNet34 고해상도 (restart61) | `train_route/experiments/historical_scripts/` model6 → 62 → 97 → 98 → 109 → restart51 → `pairwise_margin_finetune.py` | train 70 | 기록값 |
| R2 | ResNet34 원본 해상도 (model269) | 같은 폴더, model109 → 194 → 195 → 239 → 242 → 257 → `model253_highres_scale_finetune.py` | train 70 | 기록값 |
| R3 | B4 3배 확대 fold 2 | `train_route/experiments/model326_hybrid_fullstats/KoMaP/train_ema.py` (마지막 가중치) | fold 2 학습분 | 3456 |
| R4~R6 | B4 가로 4000 fold 1·3·4 | `Model349_fusion805/model397_reproduction_bundle/model329/train.py` | 각 fold 학습분 | 0 |
| R7 | HRNet 드문 상 가중 (model347) | `train_route/experiments/model336_hrnet_best/train.py` (기본 HRNet model336 → 12 epoch 추가) | train 70 | 0, 347 |
| E1~E2 | B4 BN 고정 추가 학습 fold 1·3 | `Model349_fusion805/b4ws/train_bnfreeze.py` (R4~R6과 같은 fold 모델에서 2,000 step) | 각 fold 학습분 | 819349 |
| E3 | HRNet 세 밝기 입력 (N3) | `model336_hrnet_best/train.py` (model336 → 25 epoch, 입력은 `tools/make_zcdf_png.py`) | train 70 | 414 |
| E4 | HRNet 가로 폭 정규화 seed1 fold4 | `model397_reproduction_bundle/model392/train.py` | fold 4 학습분 | 1 |
| E5 | HRNet 가로 폭 정규화 70장 전체 | 같은 코드 | train 70 | 5 |
| F1 | 3층 CNN 보정기 | `refiner_retrain/build_features.py` → `train_refiner.py` | E의 train 70 OOF 예측 | 81349 |

- **fold:** 학습용 70장을 시편(A번호) 단위로 4묶음으로 고정 분할합니다(`model397_reproduction_bundle/config/model397_group_folds.json`).
- **F 학습 입력:** F는 학습에 쓰지 않은 fold 모델의 예측(OOF)으로 학습하므로, B4 가로 4000 fold 1~4와 HRNet seed1·seed2 fold 1~4를 모두 학습합니다. 추론에는 위 13개만 씁니다.

## 5. 재현 범위와 원래 학습과의 차이

- **GPU 연산 비결정성:** 같은 코드·시드라도 가중치가 비트 단위로 같지 않아, 재학습한 모델의 점수는 80.78%에서 조금 달라질 수 있습니다. 결합 비율·후처리 설정은 위 재조정 값으로 고정이며, 재학습 뒤 다시 고르지 않습니다.
- **valid 사용 범위:**
  - R1·R2의 ResNet34 이어 학습은 원래 파이프라인대로 단계마다 best.pt를 검증용 점수로 고릅니다. 학습 자체에는 검증용을 쓰지 않습니다.
  - 나머지 모델은 마지막 가중치를 쓰며 검증용을 보지 않습니다.
- **R4~R6과 E1~E2의 부모 모델:** 원래는 같은 레시피·같은 fold로 따로 학습한 두 벌을 썼습니다. 여기서는 한 번 학습해 R과 E가 공유합니다.
- **model336 학습 중 검증 로그:** 원래는 5 epoch마다 검증용 점수를 기록만 했고, 여기서는 기록하지 않습니다(`--val-every 0`).
- **`inference/parent_metadata/`:** 공개 가중치와 함께 만든 학습 기록(args.json)입니다. 원래 학습 서버 경로가 적혀 있지만 기록일 뿐, 코드가 그 경로를 읽지 않습니다.

## 6. 이 패키지로 확인한 것 (anacnu, RTX 5080)

- **공개 가중치 예측:** `run_predict.sh`로 검증용 20장과 평가용 10장을 원본 이미지부터 예측했습니다.
  - 검증용 mIoU는 80.779378%입니다. 저장 확률로 계산한 재조정 점수(80.779341%)와 0.00004%p 차이입니다.
  - 저장 확률 재조정 마스크와 비교하면 검증용 16픽셀(15,757,260 중), 평가용 14픽셀(9,074,964 중)만 다릅니다. 저장 확률 쪽이 B4 확률을 FP16으로 평균해 둔 값이라 생기는 반올림 차이입니다.
  - 같은 패키지의 원래 설정(lite13)으로는 lite13 최종 마스크와 차이 0픽셀, 80.720984%를 확인했습니다.
- **재학습 흐름:** `SMOKE=1 bash run_retrain.sh 0`으로 13개 가중치 학습, F 학습, 가중치 배치, 검증용 채점, 평가용 마스크 10장 생성까지 오류 없이 끝나는 것을 확인했습니다. 반복 수를 줄인 시험이라 점수는 의미가 없습니다.
- **분할과 입력:**
  - 학습용 4-fold 분할은 원래 학습에 쓴 fold 구성과 같습니다.
  - N3 입력 이미지(`tools/make_zcdf_png.py`)는 원래 학습 입력과 70장 모두 같습니다.
- **미확인:** 전체 반복 수로 한 재학습(약 80 GPU시간)은 아직 끝나지 않았습니다. 같은 학습 코드의 lite13 패키지로 진행 중입니다.
- **GPU 점유 확인:** 공유 서버에서 다른 사용자가 쓰는 GPU에서 시작하지 않으려면 `KOMAP_REQUIRE_IDLE_GPU=1`을 설정합니다.
