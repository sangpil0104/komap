# Model349 보고서 추가 자료

**PNG 그림18개와 Markdown 표·해설11개**를 모은 자료이다. Model349 원래 점수는 validation20 image mIoU **80.068799%**이다.

데이터 통계는 제공된 train70/valid20/test10에서 계산했고, test 클래스 마스크는 사용하지 않았다. 추가 비교는 저장된 validation 확률로만 계산했다. 기존 valid/test 마스크·comparison 및 기존 상세 보고서는 중복으로 담지 않았다.

**그림 목록**

| 파일 | 내용 | 권장 위치 |
| --- | --- | --- |
| [01_dataset_overview.png](figures/01_dataset_overview.png) | 데이터 구성 | 1-3 EDA |
| [02_image_resolution.png](figures/02_image_resolution.png) | 해상도 분포 | 1-3 EDA |
| [03_class_pixel_share.png](figures/03_class_pixel_share.png) | 클래스별 픽셀 구성비 | 1-3 EDA |
| [04_class_image_area.png](figures/04_class_image_area.png) | 이미지별 면적 비율 | 1-3 EDA |
| [05_brightness_distribution.png](figures/05_brightness_distribution.png) | 밝기 및 대비 분포 | 1-3 EDA |
| [06_specimen_overlap.png](figures/06_specimen_overlap.png) | 시편 ID 중복 | 1-3 EDA / 평가 조건 |
| [07_raw_zscore_cdf.png](figures/07_raw_zscore_cdf.png) | Raw/Z-score/CDF 입력 예시 | 2-1 전처리 |
| [08_preprocessing_flow.png](figures/08_preprocessing_flow.png) | 전처리·추론 흐름 | 2-2 전처리 절차 |
| [09_model349_pipeline.png](figures/09_model349_pipeline.png) | Model349 전체 구조 | 3-2 모델링 방법 |
| [10_resnet_aspp_native.png](figures/10_resnet_aspp_native.png) | ResNet34·ASPP·native stem | 3-1 알고리즘 / 3-2 모델링 |
| [11_b4_hrnet_architecture.png](figures/11_b4_hrnet_architecture.png) | B4·HRNet 전문가 구조 | 3-1 알고리즘 / 3-2 모델링 |
| [12_classwise_fusion.png](figures/12_classwise_fusion.png) | 클래스별 결합 계수 | 3-2 모델링 방법 |
| [13_stage_performance.png](figures/13_stage_performance.png) | 단계별 성능 | 3-3 성능 |
| [14_class_iou.png](figures/14_class_iou.png) | 클래스별 IoU | 3-3 성능 |
| [15_per_image_miou.png](figures/15_per_image_miou.png) | 이미지별 mIoU | 3-3 성능 |
| [16_per_image_class_iou.png](figures/16_per_image_class_iou.png) | 이미지별 클래스 IoU heatmap | 3-3 성능 |
| [17_frozen_cache_ablation.png](figures/17_frozen_cache_ablation.png) | 저장 확률 기반 후처리·결합 비교 | 3-3 성능 / 추가 분석 |
| [18_image_miou_definition.png](figures/18_image_miou_definition.png) | 평가 지표 설명 | 3-3 성능 |

**표 목록**

| 파일 | 내용 | 권장 위치 |
| --- | --- | --- |
| [01_dataset_summary.md](tables/01_dataset_summary.md) | 데이터 구성 및 분할 | 1-3 EDA |
| [02_class_distribution.md](tables/02_class_distribution.md) | 클래스 불균형과 이미지별 면적 분포 | 1-3 EDA |
| [03_preprocessing_and_augmentation.md](tables/03_preprocessing_and_augmentation.md) | 입력 표현·전처리·증강 | 2-1 전처리 방법 / 2-2 절차 |
| [04_model_components.md](tables/04_model_components.md) | 모델 구성과 설계 의도 | 3-1 배경/알고리즘 / 3-2 모델링 |
| [05_training_and_inference.md](tables/05_training_and_inference.md) | 학습·추론 설정 | 3-2 모델링 방법 |
| [06_fusion_and_postprocessing.md](tables/06_fusion_and_postprocessing.md) | 클래스별 결합과 후처리 | 3-2 모델링 방법 |
| [07_stage_performance.md](tables/07_stage_performance.md) | 조합 단계별 성능 | 3-3 성능 |
| [08_validation_scores.md](tables/08_validation_scores.md) | Validation 이미지별·클래스별 점수 | 3-3 성능 |
| [09_frozen_probability_comparisons.md](tables/09_frozen_probability_comparisons.md) | 저장 확률 기반 후처리·결합 비교 | 3-3 성능 / 추가 분석 |
| [10_evaluation_and_reproducibility.md](tables/10_evaluation_and_reproducibility.md) | 평가 조건·재현 정보·제출 산출물 | 3-2 방법 / 3-3 성능 / 재현 정보 |
| [11_figure_captions_and_page_plan.md](tables/11_figure_captions_and_page_plan.md) | 그림 캡션 및 10페이지 배치안 | 작성 가이드 |

그림별 캡션과10페이지 배치안은 [표11](tables/11_figure_captions_and_page_plan.md)에 있다. 수치표의 단위(%, %p)와 전체 픽셀 구성비/이미지별 평균의 차이를 유지하여 인용한다.

PNG는300dpi이며 한글 글꼴은NanumGothic을 사용했다. 그림18의60%는 지표 설명용 예시이다. 새로 학습한 모델의 결과를 포함하지 않는다.

저장확률 비교에서 공간 필터 제거가 +0.000135%p 높게 나타난 사실도 그대로 수록했다. 원래 모델 점수나 checkpoint를 바꾸지 않았다.

분석·생성 코드: `experiments/model349_report_materials_v1/analyze.py`, `figures.py`, `tables_and_package.py`. 모든 스크립트는 Conda `komap`에서 실행했다.
