# Model349 exh14 — 구성 가중치 14개 재학습, GPU 2~7

사용자가 요청한 **exh14 구성 모델 재학습** 실험입니다.
가중치 14개를 각각 한 번 새로 학습한 뒤, GitHub exh14의 고정 추론 코드로 valid20을 평가합니다.
모든 모델이 완료되어야 최종 exh14를 평가하며 기존 가중치로 빈 구성을 채우지 않습니다.

## 직접 시작

```bash
cd /Data3/sppark/komap/experiments/model349_exh14_retrain_gpu234567_v1
./start_nohup.sh
```

위 명령을 **사용자가 실행해야 학습이 시작됩니다**. conda `komap`을 자동 활성화합니다.
터미널 직접 실행은 `./run.sh train --execute`입니다.
출력되는 PID·로그 경로와 `tail -f ...` 명령으로 진행률·경과 시간·ETA를 확인합니다.
구성 모델별 epoch/iteration tqdm과 상세 로그는 실행 폴더의 `logs/`에 있습니다.

물리 GPU **2,3,4,5,6,7**을 UUID로 지정하고, 모델/fold 하나에 GPU 하나씩 최대 6개를 병렬 배정합니다.
부모 모델이 먼저 끝나야 해당 자식을 시작합니다. DDP는 사용하지 않습니다.
GPU가 사용 중이면 비기를 기다립니다(메모리 64 MiB 이하, 사용률 0일 때 새 작업 배정).
웹 서버나 대시보드는 사용하지 않습니다. 추정 ETA에 GPU 대기와 최종 추론·저장 시간은 포함되지 않습니다.

중단은 출력된 PID에 `kill -TERM PID`를 사용합니다. 이 실행의 학습 자식·데이터 로더도 종료합니다.
실패·중단된 출력은 남기며, 새 실행은 항상 새 폴더에서 14개를 다시 시작합니다. 자동 재개는 없습니다.
동일 실행기의 동시 중복 실행은 잠금으로 차단합니다.

학습 없이 확인:

```bash
./run.sh plan
./run.sh check
./run.sh train  # --execute 없이는 사전 점검만 수행
```

## 재학습 구성

| 구성 | 수 | 학습량 | 초기 가중치 / 새 부모 | seed |
|---|---:|---|---|---|
| Model109 ASPP | 1 | 4 epochs, 512 patch | 보존된 Model98 | 42 |
| restart51 | 1 | 6 epochs, 768 patch | 이번에 학습한 Model109 | 123 |
| restart59 | 1 | 5 epochs, 768 patch | 이번에 학습한 Model109 | 456 |
| restart61 pairwise | 1 | 4 epochs, 768 patch | 이번에 학습한 restart51 | 42 |
| Model269 native | 1 | 4 epochs, 1024 patch | 보존된 Model257 | 269 |
| B4 322 | 2 | fold 3·4 각각 16,000 iterations | ImageNet encoder | 각 0 |
| B4 326 | 2 | fold 1·2 각각 16,000 iterations | ImageNet encoder | 각 0 |
| B4 329 | 3 | fold 1·3·4 각각 16,000 iterations | ImageNet encoder | 각 0 |
| HRNet336 | 1 | 70 epochs, batch 4 × accumulation 2 | ImageNet encoder | 0 |
| HRNet347 | 1 | 12 epochs, 역빈도 CE 미세조정 | 이번에 학습한 HRNet336 | 347 |
| 합계 | **14** | | | |

Model98·257은 이번 범위 밖의 선행 모델이므로 보존된 부모 가중치 사본을 사용합니다.
전체 선행 계보를 처음부터 학습하거나 100개 시드를 탐색하는 실험은 아닙니다.
epoch·iteration·학습률·손실·증강·checkpoint 선택은 기존 Model349 학습 레시피를 유지했습니다.
restart61의 원래 학습 내부 calibration 기반 epoch 선택도 유지하되, 이 calibration을
최종 exh14의 고정 결합·후처리에 적용하지 않습니다.

고해상도 모델은 `best.pt`, HRNet·B4는 `last.pt`를 사용합니다.
학습 완료 후 추론용 사본에는 **FP32 텐서를 그대로** 담으며 FP16 저장 변환을 하지 않습니다.
원래 추론의 autocast와 중간 확률 float16 저장은 유지합니다.

## 데이터와 지표

학습 70장 / 검증 20장 및 대응 정답을 새 실험 경로에 복사했습니다.
B4는 기존 specimen group 4-fold 분할을 유지하며 필요한 fold만 학습합니다.
각 fold의 train/heldout은 시편 기준으로 겹치지 않고, 공식 valid20은 fold 학습에 포함하지 않습니다.
HRNet·고해상도 모델의 원래 검증/epoch 선택 방식은 유지합니다.

지표는 **이미지별 네 클래스 IoU 평균 → valid 20장 평균**입니다.
정답·예측 모두 없는 클래스 IoU=0이며 global mIoU는 계산하지 않습니다.
최종 결과는 새 가중치 14개를 모두 결합한 exh14의 점수입니다.
HRNet 학습 중 빠른 검증값 등 개별 학습 로그는 최종 exh14 성능과 구분합니다.

exh14의 fold·gamma·밝기 라우팅·calibration·후처리는 고정합니다.
고해상도 경로만 D4 TTA와 이미지별 TENT 5단계를 사용하며, B4/HRNet/Native는
원본 실행 코드의 TTA off를 유지합니다. TENT는 저장 체크포인트를 변경하지 않습니다.
원래 exh14 조합은 valid20으로 선택했으므로 이 점수는 독립 test 성능이 아닙니다.
기존 모델이나 GitHub 공개 점수를 이번 재학습 결과로 사용하지 않습니다.

## 원본과 결과 보존

- GitHub `CNUKoMaP/lhs` commit `1704fff772a713101a2d63467b4d9321420ada3f`의
  `Model349_exh14_code.zip`, `Model349_train_route_code.zip`을 `downloads/`에 보존했습니다.
- `frozen/`: 원본 exh14 추론 코드·설정. `upstream_training/`: GitHub 학습 소스 원본.
- `source/`: 기존 image-mIoU 학습 소스를 별도로 복사한 실행 사본.
  이전 소스/모델/결과를 변경하지 않았습니다. 원본과 차이는 `checks/training_source_patches.diff`입니다.
  기존 tqdm·image mIoU 수정과 worker RNG 고정, 불필요한 test 추론 생략을 포함합니다.
- B4 worker의 시간 기반 RNG를 끄고 NumPy/Python/torch/Albumentations 시드를 고정합니다.
  GPU 커널의 비트 단위 결정성까지 강제하지는 않습니다.
- `initial_weights/`: 부모 Model98·257의 별도 사본. `data/`, `folds/`: 새 실험의 입력.
- `outputs/exh14_train_한국시간_고유ID/`: 실행 소스 스냅샷, 환경, 계획, 명령/이벤트,
  모든 학습 체크포인트, 최종 추론용 14개 모델, 중간 확률, 마스크, 학습/추론 로그.

학습과 최종 검증이 완료되면 한국 날짜 기준
**`/Data3/sppark/komap/results/model_results_YYYYMMDD_다음번호/`**를 새로 만듭니다.

- 같은 이름의 Markdown 보고서, `metrics.json`, `per_image.csv`.
- `valid_mask_comparison/`에 검증 20장의 2×2 이미지.
  좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth.
  맞은 픽셀은 흰색, 틀린 픽셀은 정답 색. 상단에 image mIoU와 100%−image mIoU 표시.
- `selected_model/`: **새 가중치 14개**, 원본 exh14 추론 소스·고정 설정, 새로운 해시 manifest.
- `source_snapshot/`: 실행 당시 학습 코드와 실행기. 입력 데이터·선행 모델은 실험 폴더에 보존합니다.
- `training_events.jsonl`, `environment.freeze.txt`, 출처와 입력 해시.
- `complete.json`이 있어야 결과 저장이 완료된 것입니다.

`selected_model/verify_bundle.py`는 **새 재학습 가중치의 해시**를 검사합니다.
원본 manifest는 `upstream_checkpoint_manifest.json`, `upstream_recipe_manifest.json`으로 보존합니다.

사전 점검 기록은 `checks/`와 `BUILD_REPORT.md`를 참조하세요.
패키지 설치/업그레이드와 학습/실제 추론은 준비 단계에서 실행하지 않습니다.
