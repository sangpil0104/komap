# KoMaP 2026 Process 1 — Al 주조합금 미세조직 4상 분할

고탄성 Al 주조합금의 광학현미경 이미지를 **Al(모재), Al3Ni, Primary Si, Eutectic Si** 네 상으로
픽셀 단위 분할하는 모델입니다. [KoMaP AI Competition 2026 Process 1](https://komap.ai/front/competition2026/process1)
과제용으로 개발했고, 이 저장소는 최고 단일 모델을 재현하는 데 필요한 코드만 담고 있습니다.

## 결과

valid 20장, 원본 해상도, overlap-tile + flip TTA 기준 (평가 지표: 4클래스 mIoU)

| | Al | Al3Ni | Primary Si | Eutectic Si | **mIoU** |
|---|---|---|---|---|---|
| 원래 학습 | 0.9470 | 0.7938 | 0.9026 | 0.6275 | **0.8177** |
| 이 저장소로 재학습 | 0.9470 | 0.7964 | 0.8965 | 0.6248 | **0.8162** |

- 대회 베이스라인 목표는 약 0.75입니다.
- 20장의 confusion matrix를 모두 합친 뒤 IoU를 계산한 값입니다. 이미지별 mIoU의 평균이 아닙니다.
- "이 저장소로 재학습"은 `uv sync --frozen`으로 만든 새 환경에서 기본값 그대로 `train.py`를 처음부터
  다시 돌린 결과입니다. 같은 seed여도 GPU 연산의 비결정성 때문에 결과가 조금 달라지며, 같은 설정의
  run-to-run 편차는 ±0.2~0.4%p 정도였습니다.

## 모델

| 구성 | 내용 |
|---|---|
| Decoder | UNet++ ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) |
| Encoder | HRNet-W48, ImageNet 사전학습 (timm `hrnet_w48`, smp 이름 `tu-hrnet_w48`) |
| 출력 | 4클래스 logit, 입력과 같은 해상도 |
| 보조 head | 1채널 경계 띠 예측 head. **학습 때만** 쓰고 추론에서는 사용하지 않음 |

### 학습 레시피

| 항목 | 값 |
|---|---|
| 입력 | 512×512 patch. 원본에서 256×256 영역을 잘라 2배 확대 (`input_scale=2`) |
| 샘플링 | patch의 50%는 Al3Ni 또는 Eutectic Si 픽셀 근처를 중심으로 자름 |
| 증강 | 좌우/상하 flip, 90° 회전, 스케일 0.8~1.2, 밝기·대비·감마, elastic/grid distortion, Gaussian noise/blur |
| Loss | 0.5 × CE + 0.5 × Dice (**클래스 가중치 없음**) + 1.0 × 경계 head loss (BCE + Dice) |
| 최적화 | AdamW, lr 3e-4 (encoder도 같은 lr), weight decay 1e-4, cosine, 70 epoch, batch 8 |
| 최종 모델 | 마지막 epoch 체크포인트 (`last.pt`) |

### 추론

1. 원본 이미지를 256px 타일로 자르고(타일 간 170px 겹침), 각 타일을 512px로 확대해 모델에 넣습니다.
2. 타일별 softmax 확률을 원래 크기로 줄인 뒤, 가우시안 가중치로 합쳐 원본 해상도 확률맵을 만듭니다.
3. 원본, 좌우 반전, 상하 반전, 좌우+상하 반전 4가지 결과를 평균하고 argmax합니다.

### 주요 설계 결정

- **클래스 가중치를 쓰지 않습니다.** 역빈도 가중치(Eutectic Si 1.88, Al3Ni 0.50)를 쓰면 애매한 경계 픽셀이
  희소 클래스 쪽으로 기울어서, Eutectic Si가 상 경계 밖으로 약 1px씩 번졌습니다(예측 면적 1.35배).
  가중치를 없애면 이 현상이 사라지고 mIoU가 0.792에서 0.816으로 오릅니다. 클래스 불균형은 클래스별로
  평균하는 Dice 항이 담당합니다.
- **encoder도 decoder와 같은 lr로 학습합니다.** 자연 이미지로 사전학습된 encoder를 흑백 미세조직
  이미지에 충분히 적응시키기 위해서입니다. encoder lr을 0.1배로 줄였을 때보다 mIoU가 높았습니다.
- **입력을 2배 확대합니다.** 미세한 Eutectic Si 층상 구조를 더 큰 해상도로 보게 하려는 것입니다.
  같은 연산량에서 시야는 좁아지지만, 1배나 4배보다 2배가 가장 좋았습니다.
- **경계 보조 head**는 클래스와 무관하게 "여기가 상 경계인지"를 예측하도록 같이 학습시켜, feature가 경계
  위치를 정밀하게 담도록 유도합니다. 효과는 +0.2%p 정도로 작습니다.

## 폴더 구조

```
.
├── komap_seg/
│   ├── palette.py      # 마스크 RGB 색 <-> 클래스 인덱스
│   ├── dataset.py      # 학습용 patch dataset, 평가용 원본 해상도 dataset, 증강
│   ├── model.py        # UNet++ 생성, 경계 보조 head
│   ├── losses.py       # CE + Dice, 경계 head loss
│   ├── inference.py    # overlap-tile 추론 + flip TTA
│   ├── metrics.py      # confusion matrix, IoU
│   └── checkpoint.py   # 체크포인트 저장/로드 (모델 설정 포함)
├── train.py            # 학습
├── evaluate.py         # 라벨이 있는 split의 mIoU
├── predict.py          # 마스크 PNG 생성
├── scripts/reproduce.sh
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

한 번에 학습, 평가, test 예측까지 하려면:

```bash
scripts/reproduce.sh /path/to/data runs/best
```

각 단계를 따로 실행하려면:

```bash
# 학습 (기본값이 위 레시피). 결과: runs/best/last.pt
uv run python train.py --data-root /path/to/data --output-dir runs/best

# valid mIoU
uv run python evaluate.py --checkpoint runs/best/last.pt --data-root /path/to/data --split valid

# 마스크 생성 (대회 색상 PNG, 파일명 {stem}_mask.png)
uv run python predict.py --checkpoint runs/best/last.pt \
    --images-dir /path/to/data/test/images --output-dir runs/best/test_predictions
```

- GPU는 기본으로 `cuda:0`을 씁니다. 다른 GPU는 `--device cuda:1` 또는 `CUDA_VISIBLE_DEVICES`로 지정하세요.
- `evaluate.py`와 `predict.py`의 `--checkpoint`에 파일을 여러 개 넘기면 확률맵을 평균해서 앙상블합니다.
- `--patch-size`와 `--input-scale`을 바꿔서 학습했다면 평가와 예측에도 같은 값을 넘겨야 합니다
  (기본값 512 / 2.0). 체크포인트에는 저장되지 않습니다.
- 학습 중 출력되는 `quick_val_mIoU`는 진행 상황 확인용입니다. 확대 없이 한 번에 추론한 값이라
  실제 지표보다 훨씬 낮게 나옵니다(마지막 epoch 약 0.75). 실제 성능은 `evaluate.py`로 확인하세요.

## 재현 환경과 소요 시간

| 항목 | 값 |
|---|---|
| GPU | RTX 3090 24GB 1장 |
| 학습 GPU 메모리 | 약 19.8GB (batch 8) |
| 학습 시간 | epoch당 약 80초, 70 epoch 약 95분 |
| 추론 시간 | valid 20장 평가 약 5분 (TTA 포함) |

- seed는 0으로 고정되어 있어 데이터 샘플링과 초기화 순서는 매번 같습니다. 다만 cuDNN 연산이
  비결정적이라, 같은 seed로 다시 학습해도 결과가 비트 단위로 같지는 않습니다.
- 멀티 GPU는 지원하지 않습니다.

## 체크포인트 형식

`last.pt`는 `{"model_config": {...}, "state_dict": {...}}` 형태입니다. `model_config`에는
`build_model()`에 넘길 인자(architecture, encoder_name, boundary_head)가 들어 있어서,
`komap_seg.checkpoint.load_model(path, device)`만으로 같은 구조를 다시 만들고 가중치를 불러옵니다.
학습된 가중치(약 310MB)는 저장소에 포함하지 않았습니다. 배포하려면 GitHub Releases 등에 따로 올리세요.
