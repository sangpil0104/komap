# Model349 — local image mIoU

Validation image mIoU: **80.068799%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
학습 체크포인트와 조합 선택에 valid를 사용하므로 독립 테스트 점수가 아닙니다.
원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.8751% | [image](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 79.0629% | [image](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.9877% | [image](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.4617% | [image](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 71.9084% | [image](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.6932% | [image](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 80.0571% | [image](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 82.2767% | [image](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.8919% | [image](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.2009% | [image](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.7153% | [image](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.4068% | [image](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.8070% | [image](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.5611% | [image](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.9091% | [image](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.9427% | [image](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 52.1529% | [image](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.4277% | [image](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 75.1578% | [image](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 81.8799% | [image](valid_mask_comparison/A7_step1_sample4.png) |
