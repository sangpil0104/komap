# Model349_fusion805_lite23_b4ws 전체 재학습

GitHub `CNUKoMaP/lhs`의 요청 ZIP과 같은 커밋의 통합 학습 ZIP을 내려받아 SHA-256을 확인했습니다. **물리 GPU 2,3,4,5,6,7 / conda komap / nohup**으로 실행합니다. 준비 과정에서는 학습을 시작하지 않았습니다.

```bash
cd /Data3/sppark/komap/experiments/model349_fusion805_lite23_b4ws_retrain_gpu234567_v1
./start_nohup.sh
```

실행기는 PID와 로그 경로를 출력합니다. 실시간 전체 로그:

```bash
cd /Data3/sppark/komap/experiments/model349_fusion805_lite23_b4ws_retrain_gpu234567_v1
tail -f "$(ls -t logs/train_*.log | head -n 1)"
```

학습 없이 사전 점검하거나 계획만 확인할 때:

```bash
./run.sh check
./run.sh plan
```

`./run.sh train`도 `--execute`가 없으면 사전 점검만 합니다. 터미널에서 직접 학습하려면 `./run.sh train --execute`입니다. 중단은 실행기가 출력한 메인 PID에 `kill -TERM PID`를 사용합니다. 자식 작업과 데이터 로더도 함께 정리합니다.

## 학습 구성

**기본 S=0 한 세트**이며 원본 시드를 사용합니다. 활성 편집기의 seed100 실험과는 별도입니다. 원본 레시피와 고정 결합 설정을 유지하며 새 시드 탐색이나 fold 재선택은 하지 않습니다.

| 학습 묶음 | 작업 수 | 설정 |
|---|---:|---|
| R 선행 모델 | 9 | Model6 → 62 → 97 → 98 및 194 → 195 → 239 → 242 → 257 |
| R 최종 구성, 공유 B4 제외 | 11 | 고해상도 ASPP 4, native269, B4-322 2-fold, B4-326 2-fold, HRNet336·347 |
| E의 원래 구성 | 12 | B4-329 4-fold + HRNet392 seed1·2 × 4-fold |
| E B4 BN 고정 추가 학습 | 4 | 같은 train fold, 2,000 iterations, lr 3e-5, seed 819349 |
| F 보정기 | 1 | 원래 E의 train70 OOF, seed 81349, 8 epochs / 1,120 updates |
| 합계 | **37** | OOF 추론 12 + 특징 생성 1 + valid E 추론 3을 더해 **53단계** |

공개 ImageNet encoder 초기화에서 시작합니다. 예전 실험의 학습 가중치를 시작점으로 쓰지 않으며 모든 선행 모델도 새로 학습합니다. R의 B4-329 fold 1·3·4는 E에서 한 번 학습한 원래 B4-329와 공유합니다. **BN 고정 추가 학습판은 E에만 적용**합니다.

F의 입력은 추가 학습 전 B4와 HRNet 8개에서 만든 시편 단위 OOF입니다. 추론에서 제외되는 HRNet 4개도 이 OOF를 위해 학습합니다. `full70_experiment`는 다른 미실행 후보이므로 이 계획에는 포함하지 않았습니다.

## 최종 23개와 고정 추론

- R: exh14의 14개. 밝기 배정, TENT5, fold 선택과 후처리는 요청 패키지의 `selected_R.yaml`과 동일합니다.
- E: B4ws 4개 + HRNet seed1 fold2·4 + seed2 fold1·4 = 8개. 가족별 fold 평균 후 FP16 반올림, 가중치 0.5 / 0.25 / 0.25입니다.
- F: 1개. 1단계 R+E에 F의 Al 예측을 덮어쓰는 **2단계가 요청 모델의 대표 결과**입니다. 1단계도 비교용으로 저장합니다.

총 **R14 + E8 + F1 = 23개**입니다. GitHub의 80.5464316%는 기존 고정 가중치의 개발 결과이며 새 학습 결과를 보장하지 않습니다. lite21로 바꾸지 않았습니다.

## 실행과 진행률

한 GPU에 이 실행기의 작업 하나씩, 최대 6개를 실행합니다. 물리 번호를 UUID로 변환하므로 CUDA 내부 번호와 무관하게 GPU 2~7만 사용합니다. 다른 작업이 사용하는 GPU는 비워질 때까지 기다립니다. GPU가 비면 의존성이 충족된 다음 작업을 배정합니다.

전체 로그에 `Trained=완료수/37`, 단계 진행률, 경과 시간, 예상 남은 시간과 작업별 최신 로그를 15초 간격으로 표시합니다. 각 학습 로그에도 `tqdm` 진행률·시간이 표시됩니다. ETA는 기존 RTX 4090 작업 시간과 선행 단계 추정치에 기반하며 GPU 대기, 최종 추론·저장 시간은 별도입니다. 이번 전체 재학습의 실측 시간은 아직 없습니다.

작업별 상세 로그는 `outputs/lite23_b4ws_train_실행ID/logs/작업명.log`에 있습니다. 실행마다 새 폴더를 만들며, 이 실행기는 중간 optimizer 상태 재개를 제공하지 않습니다. 다시 시작하면 새 실험으로 처음부터 실행합니다. 중단된 출력도 보존됩니다.

## 데이터, 평가, 결과

로컬에 복사한 train70 / valid20 / test10을 사용하고, R과 E의 시편 fold 일치를 점검합니다. OOF용 학습에는 train70만 사용합니다. 원본 레시피처럼 R 일부 선행/구성 모델은 valid20으로 checkpoint를 선택합니다. valid20은 독립된 최종 test가 아닙니다.

프로젝트 지침에 따라 **이미지별 4-class IoU 평균 → 이미지 평균**, empty union=0을 사용하며 global mIoU는 계산하지 않습니다. 과거 일부 R 학습기의 empty-class 제외/pooled 선택 지표는 이 기준으로 통일한 로컬 소스를 사용합니다. B4 데이터 로더의 시간 기반 난수 시드 대신 명시된 시드와 Albumentations RNG를 사용합니다. 이러한 차이와 GPU·환경 차이 때문에 원본 가중치의 비트 단위 재현은 주장하지 않습니다.

학습이 끝나면 valid20 평가와 test10 예측을 실행하고, 한국 날짜 기준 다음 빈 번호인 `results/model_results_YYYYMMDD_N/`에 저장합니다.

- `model_results_YYYYMMDD_N.md`, `metrics.json`, `per_image.csv`: 요청한 lite23 B4ws 결과.
- `valid_mask_comparison/`: valid20 각각의 2×2 비교 그림. 좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 맞은 픽셀은 흰색, 오류 픽셀은 정답 색이며 위에 image mIoU와 100%−image mIoU를 표시합니다.
- `stage1/`: F 적용 전 같은 valid20 비교 결과.
- `test_predictions/`: test10 마스크. 정답이 없어 test 점수는 계산하지 않습니다.
- `selected_model/`: 정확히 23개 가중치와 추론 코드·설정·새 SHA-256 manifest.
- `source_snapshot/`, `provenance.json`, 실행 기록: 실제 학습 소스와 출처.

선행 모델과 추론에서 제외한 OOF 모델을 포함한 모든 학습 출력은 실행 폴더에 남깁니다. 기존 모델, 소스와 결과는 수정하지 않습니다.

## 출처와 검증

- 저장소: https://github.com/CNUKoMaP/lhs
- 고정 커밋: `fdeeccf246ee8dfcecca0c9d5ebfd85f31cd18db`
- 요청 ZIP: `Model349_fusion805_lite23_b4ws_code.zip`, SHA-256 `35fb844e70f163fa133065fc0511608dd4e8afc41fba1ded7b4c6055b77729ee`
- 통합 학습 ZIP: `Model349_lite23_b4ws_bundle_code.zip`, SHA-256 `18b6e7bfd6b51b0cd0378dfd21e71d6d1441cc509538ff01367cdfff2ae4ffd4`

`downloads/`와 `upstream/`에 내려받은 원본을 보존합니다. 실행용 복사본만 수정했습니다. 검증 내역은 `BUILD_REPORT.md`와 `checks/*.json`, 로컬 소스 차이는 `checks/source_changes.diff`에 있습니다. 검증 과정에는 실제 학습이나 GPU 모델 추론이 포함되지 않습니다.
