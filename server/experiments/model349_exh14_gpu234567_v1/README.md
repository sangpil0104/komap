# Model349 exh14 — GPU 2~7 검증 실험

GitHub `CNUKoMaP/lhs`의 `Model349_exh14_code.zip`을 commit
`1704fff772a713101a2d63467b4d9321420ada3f`에서 다운로드했습니다.
`downloads/`와 `upstream/`에 배포 원본을 보존하며 `source/`의 추론 코드는 원본과 같습니다.
이 패키지는 **이미 학습된 체크포인트 14개를 사용하는 고정 추론 모델**입니다.
새 모델 학습이나 시드 탐색은 포함하지 않습니다. 준비 단계에서 실제 평가를 시작하지 않습니다.

## 시작

```bash
cd /Data3/sppark/komap/experiments/model349_exh14_gpu234567_v1
./start_nohup.sh
```

사용자가 실행하면 conda `komap`을 자동 활성화하고 valid 20장을 평가합니다.
출력되는 `tail -f ...` 명령으로 진행률·경과 시간·ETA를 확인합니다.
터미널에서 직접 실행하려면 `./run.sh evaluate --execute`를 사용합니다.

물리 GPU **2,3,4,5,6,7**을 UUID로 지정하며 GPU당 구성요소 하나, 최대 6개를 병렬 실행합니다.
전체 7개 구성요소를 실행하므로 빈 GPU에 마지막 작업을 배정합니다.
구성요소 종료 시점과 CPU 융합 단계에는 GPU 사용량이 줄어듭니다.
실행 시 요청한 GPU가 사용 중이면 중단합니다(원본 기준: 메모리 64 MiB 초과 또는 사용률 > 0).
다른 프로세스를 종료하지 않습니다. 출력된 메인 PID에 `kill -TERM PID`를 보내면
해당 실행의 추론 프로세스와 자식 작업을 종료하고 생성된 파일을 보존합니다.
동일 실행기의 동시 중복 실행은 잠금으로 차단합니다.

학습·추론 없이 확인:

```bash
./run.sh plan
./run.sh check
./run.sh check --load-models  # CPU에서 가중치 14개 엄격 로딩; forward/backward 없음
```

`./run.sh` 또는 `./run.sh evaluate`만 실행해도 사전 점검만 합니다.

## 고정 모델

| 구성 | 사용 가중치 |
|---|---|
| B4 322 | fold 3, 4 |
| B4 326 | fold 1, 2 |
| B4 329 | fold 1, 3, 4 |
| HRNet | Model336, Model347 |
| Native | Model269 |
| 고해상도 밝기 라우팅 | Model109, restart51, restart59, restart61 |

ZIP에는 가중치가 없습니다. 로컬 `results/model349_inference_bundle_20261007_1/`의
**GitHub manifest와 SHA-256이 일치하는 FP32 원본 14개(1,755,428,956 bytes)**를
`source/weights/`에 독립 파일로 복사했습니다. FP16 변환은 하지 않았습니다.
B4 로더에 필요한 `args.json` 7개도 같은 기존 릴리스에서 복사하고 해시를 기록했습니다.
기존 모델, 소스, 결과는 수정하지 않았습니다.

원본 결합 비율·fold 선택·후처리를 그대로 사용합니다.
고해상도 경로는 D4 TTA와 이미지별 **TENT 5단계**를 사용합니다.
TENT는 정답 없이 해당 이미지의 BN 파라미터만 메모리에서 적응시키며 체크포인트 파일을 수정하지 않습니다.
B4/HRNet/Native269는 원본 실행 코드대로 TTA를 끕니다.
YAML의 TTA 문자열보다 실제 호출의 `tta=False`/`'none'` 설정이 우선하는 원본 동작을 유지했습니다.

`data/valid/`는 기존 Model349 검증 이미지·정답 각 20개의 별도 사본입니다.
GitHub exh14는 이 valid20으로 조합을 선택했으므로 평가 점수는 독립 test 성능이 아닙니다.
원본 README의 80.2312313%는 P100 환경에서 보고된 값이며, 로컬 실행 결과로 간주하지 않습니다.

## 저장 결과

매 실행마다 `outputs/exh14_한국시간_고유ID/`에 새 폴더를 만듭니다.
원본 실행 코드 스냅샷, 환경, 실행 명령, 검증 데이터 해시, 구성요소별 로그,
추론 확률과 마스크를 보존합니다. 재실행은 항상 새 경로이며 자동 재개하지 않습니다.

완료 시 `/Data3/sppark/komap/results/model_results_YYYYMMDD_다음번호/`를 생성합니다.
날짜는 Asia/Seoul 기준이고 기존 폴더를 덮어쓰지 않습니다.

- 같은 이름의 Markdown 보고서, `metrics.json`, `per_image.csv`.
- `predicted_labels/`, `predicted_masks/`.
- `valid_mask_comparison/`에 검증 20장의 2×2 비교 이미지.
  좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth.
  맞은 픽셀은 흰색, 틀린 픽셀은 해당 위치의 정답 색으로 표시합니다.
  상단에 image mIoU와 100%−image mIoU를 표시합니다.
- `source_snapshot/`에 실제 실행 당시 추론 소스·설정·가중치와 실행기 사본.
- `complete.json`이 있는 결과 폴더만 저장 완료된 결과입니다.

지표는 이미지마다 4개 클래스 IoU를 평균한 뒤 valid20에 대해 평균합니다.
정답·예측 모두 없는 클래스 IoU=0이며 global mIoU는 계산하지 않습니다.
결과 저장 시 예측 라벨로 다시 계산하여 원본 코드가 기록한 image mIoU와 대조합니다.

추론이 완료됐지만 보고서 저장만 실패했다면 새 보고서 폴더로 재저장할 수 있습니다.

```bash
./run.sh export --run /Data3/sppark/komap/experiments/model349_exh14_gpu234567_v1/outputs/exh14_실행ID
```

환경 패키지를 설치·업그레이드하지 않고 기존 conda `komap`으로 점검합니다.
실제 점검 내역은 `checks/`와 `BUILD_REPORT.md`에 기록합니다.
