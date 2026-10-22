# Model349 — local image mIoU

Validation image mIoU: **12.608992%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
학습 체크포인트와 조합 선택에 valid를 사용하므로 독립 테스트 점수가 아닙니다.
원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model349_train_route_v1/outputs/checks/routes_9s4ahn1g`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A10_step1_sample1 | 12.8463% | [image](valid_mask_comparison/A10_step1_sample1.png) |
| A11_step1_sample1 | 10.9097% | [image](valid_mask_comparison/A11_step1_sample1.png) |
| A12_step1_sample1 | 11.1205% | [image](valid_mask_comparison/A12_step1_sample1.png) |
| A13_step1_sample1 | 13.1224% | [image](valid_mask_comparison/A13_step1_sample1.png) |
| A14_step1_sample1 | 13.9154% | [image](valid_mask_comparison/A14_step1_sample1.png) |
| A15_step1_sample1 | 13.4305% | [image](valid_mask_comparison/A15_step1_sample1.png) |
| A16_step1_sample1 | 13.3692% | [image](valid_mask_comparison/A16_step1_sample1.png) |
| A17_step1_sample1 | 12.5523% | [image](valid_mask_comparison/A17_step1_sample1.png) |
| A18_step1_sample1 | 12.8594% | [image](valid_mask_comparison/A18_step1_sample1.png) |
| A19_step1_sample1 | 13.2077% | [image](valid_mask_comparison/A19_step1_sample1.png) |
| A1_step1_sample1 | 13.2331% | [image](valid_mask_comparison/A1_step1_sample1.png) |
| A20_step1_sample1 | 11.1902% | [image](valid_mask_comparison/A20_step1_sample1.png) |
| A2_step1_sample1 | 12.6365% | [image](valid_mask_comparison/A2_step1_sample1.png) |
| A3_step1_sample1 | 12.5098% | [image](valid_mask_comparison/A3_step1_sample1.png) |
| A4_step1_sample1 | 12.2445% | [image](valid_mask_comparison/A4_step1_sample1.png) |
| A5_step1_sample1 | 12.9542% | [image](valid_mask_comparison/A5_step1_sample1.png) |
| A6_step1_sample1 | 11.3252% | [image](valid_mask_comparison/A6_step1_sample1.png) |
| A7_step1_sample1 | 13.2681% | [image](valid_mask_comparison/A7_step1_sample1.png) |
| A8_step1_sample1 | 12.9379% | [image](valid_mask_comparison/A8_step1_sample1.png) |
| A9_step1_sample1 | 12.5470% | [image](valid_mask_comparison/A9_step1_sample1.png) |
