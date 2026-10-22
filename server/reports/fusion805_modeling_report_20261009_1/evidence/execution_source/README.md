# Model349 fusion805 구성 모델 재학습 — GPU 2~7

직전 exh14 실험과 같은 실행·로그·결과 형식으로 준비한 별도 실험입니다.
R(exh14) 14개, E(B4/HRNet) 12개, F(residual) 1개를 각각 새로 학습합니다.
GitHub 최신 설명에 따라 **stage1 R+E를 주 결과**, stage2 R+E+F를 비교 결과로 함께 저장합니다.
원래 공개된 점수를 새 가중치의 성능으로 사용하지 않습니다.

## 직접 시작

```bash
cd /Data3/sppark/komap/experiments/model349_fusion805_retrain_gpu234567_v1
./start_nohup.sh
```

위 명령을 **사용자가 실행해야 학습이 시작됩니다**. conda `komap`을 자동 활성화합니다.
터미널 직접 실행은 `./run.sh train --execute`입니다.
물리 GPU **2,3,4,5,6,7**을 UUID로 지정하고 작업당 GPU 하나, 최대 6개를 병렬 배정합니다.
부모 작업 완료 후 다음 작업을 배정하며 GPU가 사용 중이면 비기를 기다립니다.
학습 27개와 OOF·검증 추론/전처리 16개, 총 43개 단계 후 최종 결합·채점·저장을 수행합니다.
새 작업 배정 기준은 해당 GPU 메모리 사용 64 MiB 이하·사용률 0입니다.

전체 실시간 로그:

```bash
tail -f "$(ls -t logs/train_*.log | head -n 1)"
```

전체 단계 tqdm, 경과 시간, 예상 남은 시간과 `Trained=n/27`이 표시됩니다.
ETA는 계획에 기록한 추정치이며 GPU 대기 및 최종 추론·저장 시간은 제외합니다.
구성 모델별 tqdm/상세 로그는 시작 로그의 `RUN_DIRECTORY` 아래 `logs/`에 저장됩니다.
웹 서버나 브라우저 대시보드는 사용하지 않습니다.
중단은 출력된 PID에 `kill -TERM PID`입니다. 해당 실행의 자식 프로세스도 종료합니다.
실패·중단된 출력은 보존하며 다음 실행은 새 폴더에서 처음부터 시작합니다. 자동 재개는 없습니다.
동일 실행기의 동시 중복 실행은 잠금으로 차단합니다.

학습 없이 확인:

```bash
./run.sh plan
./run.sh check
./run.sh train  # --execute 없으면 사전 점검만 수행
```

## 고정 학습 구성

| 구성 | 개수 | 설정 |
|---|---:|---|
| R: exh14 | 14 | 직전 실험과 같은 학습 소스·부모 계보·seed·epoch/iteration·고정 추론 |
| E: Model329 B4 | 4 | specimen 4-fold, 각 16,000 iterations, seed 0, batch 2, crop 1536, width 4000, zcdf3 |
| E: Model392 HRNet | 8 | specimen 4-fold × seeds 1·2, 각 70 epochs, batch 4 × accumulation 2, crop 512, scale 2, width normalization |
| F: residual | 1 | seed 81349, 8 epochs, batch 8, 1,120 updates, crop 256, 11,876 parameters, AdamW, 고정 최종 epoch |

R의 Model98·257 선행 부모는 보존된 사본으로 초기화합니다. 나머지는 새 부모 또는 ImageNet encoder에서 시작합니다.
이전 exh14 재학습 결과 가중치로 R을 대체하지 않습니다.
R의 원래 내부 검증/epoch 선택 방식은 유지합니다.
E는 마지막 checkpoint를 사용하고 학습 중 valid20으로 모델을 선택하지 않습니다.
E의 B4 worker RNG는 직전 실험처럼 시간 기반 seed 대신 고정 seed를 사용합니다.
E의 B4 D4 / HRNet flip4 TTA, family별 float32 fold 평균 → float16 반올림 → float32 결합 순서를 유지합니다.
E 결합은 B4 0.5 + HRNet seed1 0.25 + HRNet seed2 0.25입니다.

F는 **train70의 OOF 예측과 train70 정답만** 사용합니다.
각 이미지를 학습에서 제외한 fold의 B4/HRNet 예측으로 입력을 만들며 시편이 fold 사이에 겹치지 않습니다.
log(max(E, 1e-7))와 전체 이미지 gray z-score를 사용합니다.
F 손실은 0.5 CE + 0.5 Dice + 0.05 delta², 잔차 범위 ±1, gradient clip 1입니다.
원본 보정기 학습의 20분 실행 제한과 선택 GPU의 외부 compute 감지도 유지합니다.
F의 epoch·seed를 valid20 점수로 선택하지 않습니다.

stage1은 E가 Al 또는 Al3Ni로 예측한 위치에 E label, 나머지에 R label을 사용합니다.
stage2는 F가 Al로 예측한 위치에 Al, 나머지에 stage1 label을 사용합니다.
동률 처리와 클래스 순서는 GitHub 구현을 유지합니다.
GitHub의 조건부 검증에서 F 제외를 대표 구성으로 지정했으므로 새 valid 점수에 따라 주 결과를 바꾸지 않습니다.

## 데이터·지표·저장 형식

train70 / valid20 이미지·정답을 새 실험 안에 복사했습니다.
R의 원래 fold와 E의 GitHub Model397 시편 fold는 각각 보존합니다.
지표는 **이미지별 4개 클래스 IoU 평균을 구한 다음 valid20 평균**입니다.
정답·예측에 모두 없는 클래스 IoU=0이며 global mIoU는 계산하지 않습니다.
valid20은 기존 조합 개발에도 사용됐으므로 독립 test 성능을 의미하지 않습니다.

실행마다 `outputs/fusion805_train_한국시간_고유ID/`에 코드 스냅샷, 환경,
실제 명령·이벤트, 학습 체크포인트, OOF/valid 확률, 새 추론 패키지와 로그를 보존합니다.
입력·소스·분할 manifest를 시작 전에 검사합니다. 기존 실험/소스/결과를 덮어쓰지 않습니다.

학습·최종 평가 완료 후 한국 날짜 기준 새 폴더를 만듭니다:

```text
/Data3/sppark/komap/results/model_results_YYYYMMDD_다음번호/
  model_results_YYYYMMDD_다음번호.md   # stage1 주 결과와 두 단계 점수
  metrics.json / per_image.csv
  valid_mask_comparison/             # stage1, 검증 20장
  stage2/                           # stage2 md·지표·마스크·비교 이미지 20장
  selected_model/                    # 새 가중치 27개와 추론 소스·해시
  source_snapshot/                   # 실행 소스 보존
  fusion_comparison.json
  complete.json                      # 저장 완료 표시
```

비교 이미지는 좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth의 2×2입니다.
맞은 픽셀은 흰색, 틀린 픽셀은 정답 색이고 상단에 image mIoU와 100%−image mIoU를 표시합니다.
`selected_model/verify_model.py`는 새 가중치 27개 및 패키지 소스 해시를 검사합니다.

## 출처와 검증

GitHub `CNUKoMaP/lhs`, commit `d4ef4b08455a6bd849225f672fdcbaa23e24cdb1`의
`Model349_fusion805_reproduction_code.zip`을 내려받고 공개 SHA256을 확인했습니다.
`downloads/`에 원본 zip/해시/설명, `upstream/`에 압축 해제 원본을 보존했습니다.
`fusion_source/`는 실행용 사본, `frozen/`은 R 추론 원본, `source/`는 직전 R 학습 소스의 복사본입니다.
경로·진행률·image mIoU·고정 RNG와 새 F checkpoint 해시 검증을 위한 변경은
`checks/fusion_source_patches.diff`에 기록했습니다. 출처는 `provenance.json`에 있습니다.
읽기/CPU·합성 검증 기록은 `checks/`와 `BUILD_REPORT.md`에 있습니다.
준비 과정에서 실제 모델 학습이나 실제 데이터 GPU 추론은 시작하지 않습니다.
