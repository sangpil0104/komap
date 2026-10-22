# lite13 retuned 오류 수정 및 평가 재개 — GPU 3~6

**기존 학습 33개는 모두 완료됐습니다. 이 실행기는 재학습 없이 저장된 가중치 13개로 평가와 결과 저장을 마무리합니다.**

오류는 valid20 추론 완료 후 파일명 검사에서 발생했습니다.
실제 컬러 마스크 `이름_mask.png`를 검사 코드가 `이름.png`로 기대하여 누락으로 판단했습니다.
`timed_predict.py`와 `export_results.py`에서 같은 오류를 수정했고, 검사를 `prediction_checks.py`로 통일했습니다.

## 사용자 실행

```bash
cd /Data3/sppark/komap/experiments/model349_lite13_retuned_resume_gpu3456_v2
./start_nohup.sh
```

GPU **물리 3·4·5·6**, conda **komap**, **nohup**으로 실행합니다.
준비 중 학습이나 GPU 추론은 시작하지 않았습니다. 시작 시 기존 입력·가중치 해시와 GPU 점유를 확인합니다.

실시간 로그:

```bash
tail -f /Data3/sppark/komap/experiments/model349_lite13_retuned_resume_gpu3456_v2/logs/latest.log
```

사전 점검은 `./run.sh check`, 재개 계획은 `./run.sh plan`입니다.
`./run.sh resume`은 점검만 수행하며, 직접 실행하려면 `./run.sh resume --execute`입니다.
`latest_run.json`에는 이번 실행 폴더, `logs/latest.pid`에는 PID가 기록됩니다.
중단할 때 `kill -TERM "$(cat logs/latest.pid)"`를 실행하면 하위 추론 프로세스도 종료합니다.

## 재개 범위

1. 기존 33개 학습과 48개 완료 단계의 산출물 해시 검증.
2. 저장된 동일 가중치로 valid20 새 추론 및 시간 측정.
3. test10 추론 및 시간 측정.
4. valid20 image mIoU 채점, 비교 그림 20장, 결과 보고서 및 최종 모델 패키지 저장.

원래 valid20 라벨과 컬러 마스크는 모두 정상입니다. 하지만 전체 추론시간 JSON을 쓰기 전에 오류가 발생했으므로,
정확한 전체 시간을 기록하기 위해 valid만 다시 추론합니다. 원래 R/E/F 단계별 시간은 보존하고, 전체 실측 시간으로 대신 사용하지 않습니다.
기존 valid20 결과로 독립 계산한 image mIoU는 **80.471931%**입니다. 최종 보고서는 새 평가 결과를 기록합니다.

원래 valid R/E/F 단계별 시간 합은 약 405초입니다. 새 valid/test 추론과 저장까지 대략 10~15분을 예상하지만
입력 해시 검사, 서버 부하와 입출력에 따라 달라질 수 있습니다. 터미널에는 단계 진행률, 경과 시간과 추정 남은 시간을 표시합니다.

`source/`의 학습·추론 코드는 실제 원래 실행의 소스 사본과 바이트 단위로 동일합니다.
학습 레시피, 최종 가중치, retuned 설정, 예측 파일 이름은 바꾸지 않았습니다.
재개 프로그램에는 학습 실행 경로가 없습니다. 원래 실험의 소스·가중치·로그·예측 결과를 덮어쓰지 않습니다.
매번 실행 시 이 폴더 아래 새로운 `outputs/resume_실행ID/`를 생성합니다.

## 시간과 결과 저장

학습 완료 모델의 새 추론에서 모델 로딩, 전처리, R(TTA/TENT)/E/F, 후처리, 확률·마스크 저장과 임시 폴더 정리까지 측정합니다.
학습·정답 채점·결과 그림 생성은 추론시간에서 제외합니다. 장당 시간은 전체 시간/장수이며 단일 이미지 지연시간이 아닙니다.
이제 시간 기록을 출력 검사보다 먼저 저장하여, 이후 검사가 실패해도 측정 시간이 남습니다.

완료 시 한국 날짜 기준 새 `results/model_results_YYYYMMDD_번호/`에 저장합니다.

- 결과 MD, `metrics.json`, `per_image.csv`.
- `inference_timing.json`, `.csv`: valid20/test10 전체 시간·장당 평균과 R/E/F별 시간.
- `valid_mask_comparison/`: 원본/정답색 오류/예측/정답의 2×2 그림 20장, image mIoU와 100%−image mIoU 표시.
- `selected_model/`: 기존 학습에서 검증한 가중치 13개와 추론 코드·해시 검증 실행기.
- `test_predictions/`: test10 예측 마스크. 정답이 없어 test 점수는 계산하지 않습니다.
- `source_snapshot/`, `stage_receipts/`: 이번 실행 소스 및 원래 완료 단계 기록.

image mIoU는 이미지별 4개 클래스 IoU 평균의 이미지 평균이며 빈 클래스 IoU는 0입니다.
기존 결과·모델을 확인한 CPU 검사로 실제 새 GPU 추론 완료를 주장하지 않습니다.

검증: `checks/parent_audit.json`(550개 입력 파일), `checks/test_resume.log`(5개 회귀 검사),
`checks/export_check.json`(실제 기존 valid20 파일로 결과 저장 검증), `checks/preflight.json`.
원래 실행 위치와 오류 분석은 `parent.json`, `ERROR_ANALYSIS.md`에 기록했습니다.
