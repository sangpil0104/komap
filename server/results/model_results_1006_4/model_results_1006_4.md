# Model364 — local image mIoU

Validation image mIoU: **79.915894%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
일부 선행 체크포인트 선택과 원본 조합 개발에 valid를 사용했습니다. 이번 최종 조합은 기록된 고정 설정이며 독립 테스트 점수가 아닙니다.
원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다. 원본 기록 점수는 참고값이며 새 모델의 점수가 아닙니다.
이번 재개 실행은 Model361 scale3 추론 overlap을 170→114로 수정했습니다(native_tile171, stride57). 원본 추론 인자와 다른 로컬 설정입니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model364_resume_gpu4567_v1/outputs/runs/model364_resume_20261006T101418_491448Z`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A117_step4_sample2 | 86.6419% | [image](valid_mask_comparison/A117_step4_sample2.png) |
| A11_step1_sample4 | 78.7087% | [image](valid_mask_comparison/A11_step1_sample4.png) |
| A11_step2_sample4 | 76.2755% | [image](valid_mask_comparison/A11_step2_sample4.png) |
| A125_step1_sample4 | 79.1854% | [image](valid_mask_comparison/A125_step1_sample4.png) |
| A153_step3_sample3 | 73.2721% | [image](valid_mask_comparison/A153_step3_sample3.png) |
| A170_step4_sample3 | 88.8828% | [image](valid_mask_comparison/A170_step4_sample3.png) |
| A181_step1_sample1 | 80.2640% | [image](valid_mask_comparison/A181_step1_sample1.png) |
| A186_step2_sample2 | 80.3903% | [image](valid_mask_comparison/A186_step2_sample2.png) |
| A191_step4_sample3 | 81.7115% | [image](valid_mask_comparison/A191_step4_sample3.png) |
| A40_step3_sample3 | 82.6550% | [image](valid_mask_comparison/A40_step3_sample3.png) |
| A42_step2_sample4 | 83.0859% | [image](valid_mask_comparison/A42_step2_sample4.png) |
| A44_step4_sample3 | 89.9096% | [image](valid_mask_comparison/A44_step4_sample3.png) |
| A55_step2_sample1 | 81.5640% | [image](valid_mask_comparison/A55_step2_sample1.png) |
| A59_step3_sample1 | 82.6281% | [image](valid_mask_comparison/A59_step3_sample1.png) |
| A63_step1_sample4 | 83.7241% | [image](valid_mask_comparison/A63_step1_sample4.png) |
| A67_step1_sample1 | 74.8684% | [image](valid_mask_comparison/A67_step1_sample1.png) |
| A71_step2_sample1 | 50.8827% | [image](valid_mask_comparison/A71_step2_sample1.png) |
| A72_step4_sample4 | 86.1051% | [image](valid_mask_comparison/A72_step4_sample4.png) |
| A74_step3_sample4 | 74.9853% | [image](valid_mask_comparison/A74_step3_sample4.png) |
| A7_step1_sample4 | 82.5773% | [image](valid_mask_comparison/A7_step1_sample4.png) |
