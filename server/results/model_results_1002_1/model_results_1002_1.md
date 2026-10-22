# Model349 — no CE class weights

Validation image mIoU: **80.040537%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
학습 체크포인트와 조합 선택에 valid를 사용하므로 독립 테스트 점수가 아닙니다.
모든 선행 모델부터 CE 클래스 가중치 없이 학습했습니다. Dice·경계 손실의 비율과 샘플링은 유지했습니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model349_no_class_weights_v1/outputs/runs/model349_no_class_weights_train_20261002T112426_091302Z`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.5526% | [image](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 79.4597% | [image](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.5093% | [image](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.5221% | [image](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 72.2046% | [image](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.5685% | [image](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 79.6630% | [image](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 82.7036% | [image](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 82.4642% | [image](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.5084% | [image](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 82.7010% | [image](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 90.0546% | [image](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.8336% | [image](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.6738% | [image](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 84.1163% | [image](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 75.0703% | [image](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 51.4229% | [image](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.2324% | [image](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.1826% | [image](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.3674% | [image](valid_mask_comparison/A7_step1_sample4.png) |
