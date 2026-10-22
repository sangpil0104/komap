# Model349 lite20 B4ws 전체 재학습

**GPU: 물리 2,3,4,5,6,7 / conda: komap / 실행: nohup**. GitHub의 `Model349_lite20_b4ws_bundle_code.zip`을 별도 경로에 내려받아 준비했습니다. 기존 실험·모델·결과는 보존하며 **학습은 사용자가 시작**합니다.

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_b4ws_retrain_gpu234567_v1
./start_nohup.sh
```

실행하면 PID와 로그 경로를 출력합니다. 실시간 로그:

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_b4ws_retrain_gpu234567_v1
tail -f "$(ls -t logs/train_*.log | head -n 1)"
```

학습 전 확인:

```bash
./run.sh check
./run.sh plan
```

`./run.sh train`도 `--execute`가 없으면 점검만 합니다. 직접 터미널에서 학습하려면 `./run.sh train --execute`를 사용합니다. 중단하려면 출력된 PID에 `kill -TERM PID`를 보내면 됩니다. 자식 학습 프로세스도 정리합니다.

## lite20 설정

이전 lite23에서 R의 `model109`, `restart59(seed456)`, B4-322 fold4를 최종 추론에서 제외한 구성입니다. **R11 + E8 + F1 = 20개** 가중치를 사용합니다.

- R 고해상도 추론: restart51과 restart61(pairwise). 제외된 model109·seed456 밝기 구간은 restart51을 사용합니다. TENT5와 고정 후처리를 유지합니다.
- R B4: 322 fold3, 326 fold1·2, 329 fold1·3·4. R HRNet336·347과 native269를 유지합니다.
- E: B4-329 BN 고정 추가 학습 4-fold 평균 0.5 + HRNet392 seed1 fold2·4 평균 0.25 + seed2 fold1·4 평균 0.25.
- F: train70 OOF로 학습한 보정기. 최종 R+E+F가 대표 결과이며 R+E도 비교용으로 저장합니다.

원본 ZIP의 `run_retrain.sh`는 아직 lite23으로 평가합니다. 이 로컬 실행기는 **학습 후 valid·test 평가, 기본 R 설정, 최종 패키지와 보고서까지 lite20으로 고정**했습니다. 사용자는 위의 `start_nohup.sh`를 실행하면 됩니다.

## 학습 계획

기본 시드는 원본 **S=0 한 세트**입니다. 시드 탐색이나 조합 재선택은 하지 않습니다. 공개 ImageNet encoder 초기화에서 선행 모델부터 다시 학습합니다.

| 묶음 | 학습 수 | 내용 |
|---|---:|---|
| R 선행 모델 | 10 | Model6 → 62 → 97 → 98 → 109 및 194 → 195 → 239 → 242 → 257 |
| R 최종 구성, 공유 B4 제외 | 8 | restart51·61, native269, B4-322 fold3, B4-326 fold1·2, HRNet336·347 |
| E 원래 구성 | 12 | B4-329 4-fold + HRNet392 2 seeds × 4-fold |
| E B4 추가 학습 | 4 | 같은 fold train, BN running 통계 고정, 2,000 iterations, lr 3e-5, seed 819349 |
| F | 1 | 추가 학습 전 E의 train70 OOF, seed 81349, 8 epochs / 1,120 updates |
| 합계 | **35** | OOF 추론 12 + 특징 생성 1 + valid E 추론 3을 더해 **51단계** |

Model109는 최종 추론에서는 제외되지만 restart51 및 다른 선행 모델 학습에 필요하므로 새로 학습합니다. 사용하지 않는 restart59와 B4-322 fold4 학습은 생략합니다. R의 B4-329 fold1·3·4는 E의 원래 B4-329와 공유하며 BN 고정 추가 학습판은 E에만 사용합니다. F OOF를 위해 HRNet 8개는 모두 학습합니다.

최종 추론에 필요한 20개 외의 중간 가중치도 실행 폴더에 보존합니다. 기존 실험의 학습 가중치를 시작점으로 재사용하지 않습니다.

## GPU, 로그, 실행 보존

GPU 번호를 UUID로 변환하여 작업마다 GPU 한 개만 노출합니다. 한 GPU에 이 실행기의 작업 하나씩, 최대 6개를 배정합니다. 학습 스케줄러는 다른 작업으로 점유된 GPU가 비워질 때까지 기다립니다. ZIP에 포함된 R 추론 GPU 지정 수정도 반영했습니다.

전체 로그는 `logs/train_*.log`, 작업별 상세 로그는 `outputs/lite20_b4ws_train_실행ID/logs/작업명.log`입니다. 전체 로그에 `Trained=완료수/35`, 단계 진행률, 경과 시간과 예상 남은 시간, 작업별 최신 로그를 표시합니다. 각 학습 로그도 `tqdm`으로 진행률과 시간을 표시합니다. ETA는 추정치이며 GPU 대기와 마지막 추론·결과 저장 시간은 별도입니다.

실행할 때마다 새 폴더와 소스 스냅샷을 만듭니다. 동시 중복 실행은 lock으로 막습니다. **중간 optimizer 상태 재개는 제공하지 않으며 다시 시작하면 새 실험으로 처음부터 학습**합니다. 중단된 출력은 덮어쓰지 않습니다.

## 데이터와 결과

로컬에 복사한 train70 / valid20 / test10을 사용합니다. OOF fold는 시편 단위로 분리하고 R/E의 공유 B4 fold가 같은지 검사합니다. 원본 레시피처럼 R 일부 모델의 checkpoint 선택에는 valid20이 사용됩니다. valid20을 독립 test 성능으로 해석하지 않습니다.

평가 및 checkpoint 선택 지표는 프로젝트 지침에 맞춘 **이미지별 4-class mIoU의 이미지 평균**입니다. 정답·예측에 모두 없는 클래스는 IoU=0으로 계산합니다. global mIoU는 계산하지 않습니다. 과거 일부 R의 absent-class 제외 지표도 이 기준으로 통일했습니다. B4 worker 및 Albumentations에는 명시된 시드를 적용합니다.

학습 완료 후 한국 날짜 기준 `results/model_results_YYYYMMDD_다음번호/`를 자동 생성합니다.

- `model_results_YYYYMMDD_N.md`, `metrics.json`, `per_image.csv`: lite20 B4ws의 valid20 실측 결과.
- `valid_mask_comparison/`: valid20 각각의 2×2 비교. 좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 맞은 픽셀은 흰색, 틀린 픽셀은 정답 색이며 image mIoU와 100%−image mIoU를 표시합니다.
- `stage1/`: F 적용 전 R+E 비교 결과와 그림 20장.
- `test_predictions/`: test10 예측 마스크. 정답이 없어 test 점수는 계산하지 않습니다.
- `selected_model/`: 정확히 20개 가중치, lite20 추론 코드·설정과 새 SHA-256 manifest.
- `source_snapshot/`, 실행 기록과 `provenance.json`: 실제 사용한 소스·설정·출처.

GitHub의 **80.5301323%**는 기존 가중치의 개발 결과입니다. 이번 재학습 결과나 실행 시간을 보장하는 값이 아니며, 실제 수치는 학습 후 생성되는 보고서에 기록합니다.

## 출처와 검증 기록

- 저장소: https://github.com/CNUKoMaP/lhs
- 고정 커밋: `f47ade8ebd8842944ba86f076eac79f5a4e51ae5`
- ZIP: `Model349_lite20_b4ws_bundle_code.zip`
- SHA-256: `60cd2b9d208fb39c68cc645786a7712c96a3b18169ed483e8ee64743322b8bb8`

원본 ZIP과 해시는 `downloads/`, 원본 소스는 `upstream/`에 보존했습니다. 실행용 소스는 이전에 검증한 로컬 metric·로그 조정을 유지하면서 최신 lite20 R 추론 코드와 설정을 적용했습니다. 세부 차이는 `checks/source_changes.diff`, 검증 결과는 `BUILD_REPORT.md`와 `checks/*.json`에 있습니다. 준비 중 실제 학습이나 GPU 모델 추론은 실행하지 않습니다.
