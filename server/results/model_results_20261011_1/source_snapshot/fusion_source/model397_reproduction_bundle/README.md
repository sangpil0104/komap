# Model397 재현 번들

이 번들은 Model397의 **train-only grouped-OOF 실험**을 외부 제공 데이터로 다시 학습하고 채점하기 위한 코드와 고정 설정을 모았습니다. 데이터, 체크포인트, 예측 텐서는 포함하지 않습니다.

## 모델 구성과 기록된 결과

Model397은 서로 다른 두 계열의 OOF 확률을 고정 가중치로 평균한 조합입니다.

- Model329: EfficientNet-B4 + U-Net, 4-fold specimen-group OOF. 각 fold는 16,000 iteration 학습 후 8-way D4 TTA 추론.
- Model392: HRNet-W48 + U-Net++, 이미지 폭 정규화, CE+Dice 및 학습 전용 boundary auxiliary head. 2 seeds × 4 folds, fold별 70 epochs; 추론은 overlap-tile + flip4 TTA.
- 고정 결합: `C = 0.50 × B4 + 0.25 × HRNet seed1 + 0.25 × HRNet seed2`. 확률을 먼저 평균한 다음 argmax합니다. threshold나 route 탐색은 없습니다.
- 지표: 이미지마다 `[Al, Al3Ni, Primary Si, Eutectic Si]` 4개 클래스 IoU를 계산하고 평균한 뒤, 70개 이미지 점수를 평균합니다. 빈 union의 IoU는 0입니다.

기록된 Model397 train-OOF image-wise mIoU는 **79.9859%**입니다. 다만 이는 학습 데이터 내부의 specimen-group OOF 개발 결과이며 공식 validation 점수가 아닙니다. Model392의 사전등록 2-seed OOF gate가 실패해 Model397의 공식 valid 평가는 승인되지 않았고, full-train 모델도 만들지 않았습니다. 이 번들은 그 상태를 바꾸거나 80% 달성을 주장하지 않습니다. 재현 결과는 기록값과 차이가 날 수 있습니다.

## 데이터와 ImageNet encoder 초기값

대회에서 별도로 제공하는 `train.zip`만 사용합니다. ZIP을 풀면 다음 구조가 있어야 합니다.

```text
DATA_ROOT/
└── train/
    ├── images/*_image.png
    └── masks/*_mask.png
```

70개 이미지와 59개 specimen group이 맞는지 실행 전 검사합니다. 번들에는 `valid.zip`, `test.zip`, 이미지·mask·ImageNet 가중치가 없습니다. 파인튜닝 시작 시 `segmentation-models-pytorch`/`timm`의 `encoder_weights=imagenet` 경로가 encoder 초기값을 내려받으므로 네트워크 연결이 필요합니다. 체크포인트를 포함하지 않습니다.

## 실행

P100에서 검증한 환경은 Python 3.10.12, PyTorch 2.6.0 + CUDA 11.8입니다. 새 가상환경을 만든 뒤:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

bash scripts/run_model397_oof.sh /path/to/DATA_ROOT /path/to/model397-output
```

스크립트는 먼저 고정 4-fold split을 symlink로 준비하고, 다음 작업을 한 GPU에서 순차 수행합니다.

1. B4 4 folds 학습 및 held-out train OOF 확률 생성.
2. HRNet 2 seeds × 4 folds 학습 및 같은 held-out train images의 OOF 확률 생성.
3. 고정 확률 평균을 계산해 `results/model397_oof.json`에 공식 empty=0 image-wise mIoU와 paired group-bootstrap 비교를 기록.

각 OOF archive에는 해당 fold의 checkpoint, `args.json`, 확률 파일 및 training/inference 소스 SHA-256을 묶은 manifest를 생성합니다. 채점기는 이 provenance, 70개 ID exactly-once, fixed group fold, probability 채널/정규화를 먼저 검증한 뒤 점수를 계산합니다.

출력 루트에 체크포인트와 확률 아카이브가 생성되므로 충분한 디스크 공간을 확보하세요. 각 작업의 로그와 `args.json`도 출력 폴더에 남습니다. 완료 산출물이 있으면 그대로 보존하고 건너뜁니다. 불완전한 기존 폴더를 자동 삭제하거나 덮어쓰지 않습니다. 불완전 실행 후 재시도할 때는 기존 폴더를 보존한 채 새 출력 경로를 사용하세요.

직접 split만 준비하려면:

```bash
python scripts/prepare_oof_splits.py --data-root /path/to/DATA_ROOT --output-root /path/to/model397-output/splits
```

코드 경로, fold별 이미지·group 분리, 채널 순서, 확률 정규화, 빈 union 처리 등을 CPU smoke로 확인하려면:

```bash
python scripts/smoke_test.py
```

## 해석상 제한

- 이 파이프라인은 70장 전체에서 specimen group을 fold 밖에 유지하는 OOF 재현만 수행합니다. official valid와 test 경로를 읽지 않습니다.
- 79.9859%는 이전 실험에서 이미 노출된 OOF 결과와의 비교 기준입니다. 이 OOF 결과를 독립 test 일반화 성능으로 해석하면 안 됩니다.
- Model397의 원래 gated valid stage는 Model392 gate 실패로 미승인입니다. 이 번들에 공식 valid/test evaluator 또는 test 제출 생성기가 없는 이유입니다.
- P100에서 실행할 경우 다른 작업과 GPU를 동시에 공유하지 말고, 학과 서버 사용 규정을 따르세요.

## 번들 파일

- `model329/`: B4 학습과 train-OOF 추론에 필요한 원본.
- `model392/`: HRNet-W48/U-Net++ width-normalized 학습 및 OOF inference 원본.
- `scripts/`: 고정 fold 준비, 학습 오케스트레이션, 독립 OOF scorer, 합성 smoke.
- `config/model397_group_folds.json`: 고정 specimen-group holdout 목록.
- `reference/`: 사전등록 프로토콜과 기존 OOF/gate 결과(예측이나 체크포인트 아님).

패키지 파일 무결성은 `reference/source_sha256.txt`로 확인할 수 있습니다. 번들을 푼 디렉터리에서 `shasum -a 256 -c reference/source_sha256.txt`를 실행하세요.
