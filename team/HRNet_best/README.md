# KoMaP 2026 Process 1 — Al 주조합금 미세조직 4상 분할

고탄성 Al 주조합금의 광학현미경 이미지를 **Al(모재), Al3Ni, Primary Si, Eutectic Si** 네 상으로
픽셀 단위 분할하는 모델입니다. [KoMaP AI Competition 2026 Process 1](https://komap.ai/front/competition2026/process1)
과제용으로 개발했고, 이 저장소는 최고 단일 모델을 재현하는 데 필요한 코드만 담고 있습니다.

## 결과

평가 지표는 **image mIoU**입니다. 이미지마다 그 이미지에 나타난 클래스들의 IoU를 평균하고, 그 값을
이미지 수로 평균합니다. 모든 수치는 원본 해상도, overlap-tile + flip TTA 기준입니다.

| 평가 | Al | Al3Ni | Primary Si | Eutectic Si | **image mIoU** | global mIoU (참고) |
|---|---|---|---|---|---|---|
| 합금 단위 5-fold 교차검증 (90장, out-of-fold) | 0.9321 | 0.7919 | 0.8856 | 0.5924 | **0.7989** | 0.8068 |
| valid 20장 (train 70장으로 학습한 단일 모델) | 0.9416 | 0.7443 | 0.9100 | 0.6032 | **0.7951** | 0.8160 |

- 5-fold는 train+valid 90장을 합금 번호(`A{n}`) 단위로 나눠, 같은 합금이 학습과 평가에 동시에 들어가지
  않게 했습니다. 각 이미지는 그 이미지를 학습에 쓰지 않은 fold 모델로 한 번씩만 채점합니다.
  fold별 image mIoU는 0.7927 / 0.7847 / 0.8093 / 0.7985 / 0.8092입니다.
- 제출본은 5개 fold 모델의 확률맵을 평균한 앙상블입니다(`scripts/kfold.sh`).
- 클래스별 값은 이미지별 IoU의 평균입니다. global mIoU는 confusion matrix를 모두 합친 뒤 계산한 값으로,
  작은 이미지의 오류가 덜 반영되어 image mIoU보다 높게 나옵니다. 참고용입니다.
- 위 수치는 이 레시피를 원래 실험 코드로 학습한 체크포인트를, 이 저장소의 `evaluate.py`와
  `kfold_evaluate.py`로 다시 채점해 같은 값이 나오는 것을 확인했습니다. 이 저장소의 `train.py`로 처음부터
  다시 학습했을 때는(경계 보조 head를 켠 변형으로 확인) 원래 학습 대비 −0.2%p였습니다. 같은 설정의
  run-to-run 편차는 ±0.2~0.4%p입니다.
- 대회 베이스라인 목표는 약 0.75입니다.

## 모델

| 구성 | 내용 |
|---|---|
| Decoder | UNet++ ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) |
| Encoder | HRNet-W48, ImageNet 사전학습 (timm `hrnet_w48`, smp 이름 `tu-hrnet_w48`) |
| 출력 | 4클래스 logit, 입력과 같은 해상도 |

### 학습 레시피

| 항목 | 값 |
|---|---|
| 입력 | 512×512 patch. 원본에서 256×256 영역을 잘라 2배 확대 (`input_scale=2`) |
| 샘플링 | patch의 50%는 Al3Ni 또는 Eutectic Si 픽셀 근처를 중심으로 자름 |
| 증강 | 좌우/상하 flip, 90° 회전, 스케일 0.8~1.2, 밝기·대비·감마, elastic/grid distortion, Gaussian noise/blur |
| Loss | 0.5 × CE + 0.5 × Dice (**클래스 가중치 없음**) |
| 최적화 | AdamW, lr 3e-4 (encoder도 같은 lr), weight decay 1e-4, cosine, 70 epoch, batch 8 |
| 최종 모델 | 마지막 epoch 체크포인트 (`last.pt`) |

### 추론

1. 원본 이미지를 256px 타일로 자르고(타일 간 170px 겹침), 각 타일을 512px로 확대해 모델에 넣습니다.
2. 타일별 softmax 확률을 원래 크기로 줄인 뒤, 가우시안 가중치로 합쳐 원본 해상도 확률맵을 만듭니다.
3. 원본, 좌우 반전, 상하 반전, 좌우+상하 반전 4가지 결과를 평균하고 argmax합니다.
   체크포인트를 여러 개 주면 모델별 확률맵도 평균합니다.

### 주요 설계 결정

- **클래스 가중치를 쓰지 않습니다.** 역빈도 가중치(Eutectic Si 1.88, Al3Ni 0.50)를 쓰면 애매한 경계 픽셀이
  희소 클래스 쪽으로 기울어서, Eutectic Si가 상 경계 밖으로 약 1px씩 번졌습니다(예측 면적 1.35배).
  가중치를 없애면 이 현상이 사라지고 valid image mIoU가 0.772에서 0.795로 오릅니다.
- **Dice는 유지합니다.** Dice를 빼고 CE만 쓰면 희소 클래스 면적은 정답에 더 가까워지지만, Eutectic Si IoU가
  떨어져 image mIoU가 −0.3%p였습니다.
- **encoder도 decoder와 같은 lr로 학습합니다.** 자연 이미지로 사전학습된 encoder를 흑백 미세조직
  이미지에 충분히 적응시키기 위해서입니다. encoder lr을 0.1배로 줄였을 때보다 좋았습니다.
- **입력을 2배 확대합니다.** 미세한 Eutectic Si 층상 구조를 더 큰 해상도로 보게 하려는 것입니다.
  같은 연산량에서 시야는 좁아지지만, 1배나 4배보다 2배가 가장 좋았습니다.

### 효과가 없었던 시도

같은 레시피에 한 가지씩 더해 valid image mIoU(기준 0.7951)를 비교했습니다. 모두 noise 범위이거나 손해였습니다.

| 시도 | image mIoU |
|---|---|
| 경계 일치 Dice loss | 0.7942 |
| 경계 보조 head (`train.py --boundary-head`로 재현 가능) | 0.7934 |
| 이미지별 밝기 z-score 정규화 | 0.7932 |
| CE 단독 (Dice 제거) | 0.7920 |
| Active Boundary Loss | 0.7892 |
| 해상도 정규화 (모든 이미지를 같은 픽셀 스케일로) | 0.7887 |
| 경계 label smoothing (SVLS) | 0.7883 |

남은 오류는 대부분 상 경계 1px 주변의 흔들림이며, 675×480 이미지가 다른 해상도보다 약 4%p 낮습니다.

## 폴더 구조

```
.
├── komap_seg/
│   ├── palette.py      # 마스크 RGB 색 <-> 클래스 인덱스
│   ├── dataset.py      # 학습용 patch dataset, 평가용 원본 해상도 dataset, 증강
│   ├── model.py        # UNet++ 생성 (선택: 경계 보조 head)
│   ├── losses.py       # CE + Dice, 경계 head loss
│   ├── inference.py    # overlap-tile 추론 + flip TTA
│   ├── metrics.py      # confusion matrix, IoU
│   └── checkpoint.py   # 체크포인트 저장/로드 (모델 설정 포함)
├── train.py            # 학습
├── evaluate.py         # 라벨이 있는 split의 image mIoU (체크포인트 여러 개면 앙상블)
├── predict.py          # 마스크 PNG 생성 (체크포인트 여러 개면 앙상블)
├── make_folds.py       # 합금 단위 K-fold 분할 (symlink)
├── kfold_evaluate.py   # out-of-fold image mIoU
├── folds/alloy_to_fold.json   # 보고된 결과에 쓴 합금 -> fold 배정
├── scripts/reproduce.sh       # 단일 모델: 학습 -> valid 평가 -> test 예측
├── scripts/kfold.sh           # 5-fold: 분할 -> 학습 x5 -> OOF 평가 -> 앙상블 test 예측
├── pyproject.toml / uv.lock   # uv 환경 (버전 고정)
└── requirements.txt           # pip 환경 (같은 버전)
```

## 설치

Python 3.12, CUDA 12.6용 PyTorch 2.14 기준입니다.

**uv (권장)** — `uv.lock`으로 검증된 버전을 그대로 설치합니다.

```bash
uv sync --frozen
```

**pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

다른 CUDA 버전을 쓰려면 `pyproject.toml`의 `pytorch-cu126` index URL(pip는 `requirements.txt`의
`--extra-index-url`)을 해당 버전으로 바꾸세요.

첫 학습 때 HRNet-W48 ImageNet 가중치를 HuggingFace Hub에서 내려받습니다(약 300MB). 캐시 위치를
바꾸려면 `HF_HOME`을 지정하세요.

## 데이터

대회 데이터는 이 저장소에 포함되어 있지 않습니다. 아래 구조로 준비하고 `--data-root`로 경로를 넘깁니다.

```
<data-root>/
├── train/images/*_image.png   train/masks/*_mask.png    (70장)
├── valid/images/*_image.png   valid/masks/*_mask.png    (20장)
└── test/images/*_image.png                              (10장)
```

- 파일명은 `A{합금번호}_step{1-4}_sample{1-4}_image.png`이고, 마스크는 `_image`를 `_mask`로 바꾼 이름입니다.
- 이미지 해상도는 675×480 ~ 1440×1024로 제각각이며, 크기를 맞출 필요는 없습니다.
- 이미지는 RGBA로 저장되어 있지만 실제로는 R=G=B인 흑백입니다.

마스크 색상과 클래스 인덱스:

| index | RGB | 상 |
|---|---|---|
| 0 | (153, 127, 76) | Al (모재) |
| 1 | (25, 76, 153) | Al3Ni |
| 2 | (76, 178, 76) | Primary Si |
| 3 | (204, 204, 204) | Eutectic Si |

## 사용법

**제출본 재현 (5-fold 학습 + 교차검증 + 앙상블 예측)**, 약 8시간 30분:

```bash
scripts/kfold.sh /path/to/data runs/kfold
```

- `make_folds.py`가 `folds/alloy_to_fold.json`의 배정대로 `runs/kfold/folds/fold{0..4}`를 만듭니다
  (원본 파일을 가리키는 symlink). 다른 분할을 원하면 `--assignment none`으로 크기 균형 분할을 새로 계산합니다.
- 결과: `runs/kfold/fold{i}/last.pt`, `runs/kfold/oof.log`(교차검증 점수), `runs/kfold/test_predictions/`

**단일 모델 (train으로 학습, valid로 평가, test 예측)**, 약 1시간 40분:

```bash
scripts/reproduce.sh /path/to/data runs/best
```

각 단계를 따로 실행하려면:

```bash
# 학습 (기본값이 위 레시피). 결과: runs/best/last.pt
uv run python train.py --data-root /path/to/data --output-dir runs/best

# valid image mIoU
uv run python evaluate.py --checkpoint runs/best/last.pt --data-root /path/to/data --split valid

# 마스크 생성 (대회 색상 PNG, 파일명 {stem}_mask.png). 체크포인트를 여러 개 주면 앙상블
uv run python predict.py --checkpoint runs/best/last.pt \
    --images-dir /path/to/data/test/images --output-dir runs/best/test_predictions
```

- GPU는 기본으로 `cuda:0`을 씁니다. 다른 GPU는 `--device cuda:1` 또는 `CUDA_VISIBLE_DEVICES`로 지정하세요.
- `--patch-size`와 `--input-scale`을 바꿔서 학습했다면 평가와 예측에도 같은 값을 넘겨야 합니다
  (기본값 512 / 2.0). 체크포인트에는 저장되지 않습니다.
- 학습 중 출력되는 `quick_val_image_mIoU`는 진행 상황 확인용입니다. 확대 없이 한 번에 추론한 값이라
  실제 지표보다 훨씬 낮게 나옵니다. 실제 성능은 `evaluate.py`로 확인하세요.

## 재현 환경과 소요 시간

| 항목 | 값 |
|---|---|
| GPU | RTX 3090 24GB 1장 |
| 학습 GPU 메모리 | 약 19.8GB (batch 8) |
| 학습 시간 | epoch당 약 80초, 70 epoch 약 95분 (5-fold 전체 약 8시간 20분) |
| 추론 시간 | valid 20장 평가 약 5분 (TTA 포함) |

- seed는 0으로 고정되어 있어 데이터 샘플링과 초기화 순서는 매번 같습니다. 다만 cuDNN 연산이
  비결정적이라, 같은 seed로 다시 학습해도 결과가 비트 단위로 같지는 않습니다.
- 멀티 GPU는 지원하지 않습니다.

## 체크포인트 형식

`last.pt`는 `{"model_config": {...}, "state_dict": {...}}` 형태입니다. `model_config`에는
`build_model()`에 넘길 인자(architecture, encoder_name, boundary_head)가 들어 있어서,
`komap_seg.checkpoint.load_model(path, device)`만으로 같은 구조를 다시 만들고 가중치를 불러옵니다.
학습된 가중치(약 310MB)는 저장소에 포함하지 않았습니다. 배포하려면 GitHub Releases 등에 따로 올리세요.
