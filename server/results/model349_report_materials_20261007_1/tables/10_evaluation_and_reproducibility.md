# 평가 조건·재현 정보·제출 산출물

| 항목 | 내용 |
| --- | --- |
| Model349 실행 ID | model349_train_20260930T143930_234475Z |
| 보고된 점수 | Validation20 image mIoU 80.068799% |
| 평가 방식 | 이미지별4클래스 평균 → 20장 평균; union=0이면IoU=0 |
| 데이터 | train70, valid20, test10; 제공된 split 사용 |
| 모델 선택 | 일부 checkpoint와 최종 조합 선택에 valid 사용 |
| B4 내부 검증 | 시편 그룹 기반4-fold; 최종 valid 확률은4-fold 평균 |
| 재현 환경 | Conda komap; 실제 학습 당시 freeze는 아래 표 |
| 검증 마스크 | 기존 결과20장; 새로 만든 표의 원래 조건과 픽셀 일치 확인 |
| Test 예측 | 기존 모델·고정 조합으로10장 추론; test 정답 사용 없음 |
| 공개된 test 성능 | 정답이 없어 계산하지 않음 |
| 전체 파라미터 수 / FLOPs | 별도 계측 자료 없음 |
| 독립 test 또는 외부SOTA | 이 자료 묶음에서 검증하지 않음 |

| 라이브러리 | 기록된 버전 |
| --- | --- |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| segmentation_models_pytorch | 0.5.0 |
| timm | 1.0.29 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| opencv-python-headless | 4.13.0.92 |
| albumentations | 2.0.8 |
| pillow | 12.3.0 |
| tqdm | 4.70.1 |

보고서에서는 “본 비교 조건의 validation 성능”으로 범위를 명시한다. 외부 SOTA를 주장하려면 동일 데이터·분할·지표·모델 선택 조건의 비교 근거가 필요하다. Validation을 미사용 독립 test로 표현하지 않는다.

원본 배포 설명의79.9095189%는 과거 기록이다. 이 실행의80.068799%와 혼용하지 않는다. 새로 산출한 저장확률 비교표도 원래 모델의 보고 점수와 구분한다.

**자료 출처**

| 근거 | 프로젝트 내 위치 |
| --- | --- |
| 완료 실행 | `experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z` |
| 실행 소스 | `experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z/source_snapshot` |
| 학습 인자 | 위 실행 outputs/각 모델/args.json 및 launcher_snapshot/pipeline_tasks.json |
| 공식 로컬 결과 | `results/model_results_0930_1/metrics.json` |
| 기존 상세 설명 | `results/model349_detailed_report_20261007.md` |
| 기존 valid/test ZIP | `results/model349_valid_test_predicted_masks_20261007_1.zip` |
| 이 자료 생성 코드 | `experiments/model349_report_materials_v1/` |

이 ZIP에는 그래프·구조도와 Markdown 표만 담았다. 기존 valid/test 마스크와 비교 이미지는 사용자가 보유한 파일을 사용하면 된다. 개인정보·팀명·접수번호·수상경력은 제공되지 않아 채우지 않았다.
