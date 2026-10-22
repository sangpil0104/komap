# Model364 재현 번들

이 번들은 **Model364의 코드 및 단계별 재현 절차**를 모은 코드 전용 자료입니다. 학습 데이터, 학습된 체크포인트, validation probability archive는 포함하지 않습니다. 따라서 ZIP만으로 즉시 점수가 재현된다고 보장하지 않습니다. 대회에서 제공한 `train.zip`/`valid.zip`, 공개 ImageNet 초기 가중치 다운로드, 그리고 Model360의 선행 확률원 산출이 필요합니다.

## Model364 구성

Model364는 단일 신경망 체크포인트가 아니라 Model360의 고정 route에 Model361 HRNet-W48/U-Net++의 확률을 entropy gate로 더하는 조합입니다. route는 `[Al3Ni, Primary Si, Al, Eutectic Si]` 순서의 확률을 사용합니다.

- Model336: HRNet-W48 + U-Net++, patch 512, 입력 scale 2, 70 epochs.
- Model347: Model336에서 시작하는 weighted-CE fine-tune, 12 epochs.
- Model351: Model336 체크포인트의 D4(8-way) TTA 추론.
- Model361: Model336에서 시작하는 scale-3 fine-tune, patch 512, 12 epochs, seed 361.
- Model360: Model349 공간 보정 route에 Model351의 D4 확률을 entropy gate로 혼합.
- Model364: Model360 결과와 Model361 확률을 pixel-wise entropy gate로 혼합. 기록된 raw-valid 선택은 margin `0.0`, strength `0.25`입니다.

Model361의 재현 학습 인자는 [`config/model364_recipe.json`](config/model364_recipe.json)에 기록했고, P100에서 실제 사용한 실행 인자는 [`scripts/train_model361.sh`](scripts/train_model361.sh)에 있습니다. D4 추론과 Model364 valid 평가도 별도 스크립트로 제공합니다.

## 기록된 결과와 해석

공식 metric은 각 이미지에서 4개 class IoU를 계산하고, empty union IoU는 0으로 둔 뒤 이미지별 평균을 내고, 마지막에 이미지 평균을 냅니다.

- Model360 기준: **79.94166196%**
- Model364 raw valid 최고: **79.94297272%** (20장, +0.00131076%p)
- Group-held-out valid 선택 진단: **79.93727495%**, Model360 대비 **−0.00438701%p**

같은 20장 validation을 여러 실험에서 반복 사용했으므로 raw-valid 값은 선택 편향이 포함된 개발 점수이며 독립 test 성능이 아닙니다. Model364가 일반화 성능을 개선했다고 결론내릴 근거는 없습니다. 원본 결과 JSON은 `results/outputs/model364_model361_entropy_into_360/results.json`에 보존했습니다.

## 준비 및 실행

1. Linux/CUDA 환경에서 이 디렉터리를 프로젝트 루트에 둡니다. 프로젝트 루트에는 대회 데이터를 풀어 `data/train/{images,masks}`와 `data/valid/{images,masks}`를 준비합니다. test mask는 필요하지 않으며 ZIP에 넣지 마세요.
2. Model364의 기존 기반 route에 필요한 선행 학습/추론을 완료해 아래의 Model195/217/269/322/326/329/336/347/351/361 확률 archive와 Model183 route metadata를 만듭니다. 해당 계열의 학습·추론 코드와 이전 route 코드는 이 번들의 `experiments/` 아래에 포함되어 있지만, 선행 계열이 많아 전체 학습을 단일 명령으로 자동 실행하지는 않습니다.
3. 기록된 Model339/341/346/349/355/360 route 설정 JSON이 `outputs/`에 없으면, 기존 파일을 덮어쓰지 않는 아래 보조 명령으로 번들에 든 reference JSON을 설치합니다.

```bash
bash Model364_reproduction_bundle/scripts/install_route_references.sh /path/to/komap_project
```

4. Model336 branch를 새로 학습하고 validation probability를 생성합니다. `KOMAP_ROOT`는 데이터와 `outputs/`가 있는 프로젝트 경로입니다.

```bash
KOMAP_ROOT=/path/to/komap_project \
KOMAP_PYTHON=/path/to/python \
bash Model364_reproduction_bundle/scripts/train_model336.sh
```

Model347과 Model361은 각각 같은 Model336 checkpoint에서 출발하는 독립 fine-tune입니다. 두 branch를 생성하려면 다음을 실행합니다.

```bash
KOMAP_ROOT=/path/to/komap_project \
KOMAP_PYTHON=/path/to/python \
bash Model364_reproduction_bundle/scripts/train_model347.sh

KOMAP_ROOT=/path/to/komap_project \
KOMAP_PYTHON=/path/to/python \
bash Model364_reproduction_bundle/scripts/train_model361.sh
```

5. Model351 D4 확률이 아직 없으면 Model336 체크포인트로 생성합니다.

```bash
KOMAP_ROOT=/path/to/komap_project \
KOMAP_PYTHON=/path/to/python \
bash Model364_reproduction_bundle/scripts/infer_model351_d4_valid.sh
```

6. 필요한 선행 입력이 모두 있을 때에만 원본 Model364 grid/evaluation을 실행합니다. 이 단계는 valid 점수를 다시 산출하므로, leaderboard/test 추정치로 쓰거나 gate를 추가 튜닝하지 마세요.

```bash
KOMAP_ROOT=/path/to/komap_project \
KOMAP_PYTHON=/path/to/python \
bash Model364_reproduction_bundle/scripts/run_model364_valid_reproduction.sh
```

실행 스크립트는 파일 존재 여부만 확인하고, 누락된 선행 모델을 대신 학습하지 않습니다. 전체 선행 입력 목록과 파일 경로는 [`docs/model364_reproduction.md`](docs/model364_reproduction.md)에 있습니다. 체크포인트와 probability archive는 외부 프로젝트 `outputs/`에 생성되며 ZIP에 포함되지 않습니다.

공유 GPU 서버에서는 서버 규정에 따라 사용할 장치만 `KOMAP_VISIBLE_GPU`로 지정할 수 있습니다. 예: `KOMAP_VISIBLE_GPU=6`. 환경에 따라 `KOMAP_PYTHON`을 해당 CUDA 환경의 Python으로 지정하세요.

## 검증

아래 smoke test는 코드 구문, recipe와 reference score의 내부 일관성만 검사합니다. 학습이나 validation 재평가는 하지 않습니다.

```bash
python Model364_reproduction_bundle/scripts/smoke_test.py
```
