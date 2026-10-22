# Model349 — local image mIoU

Validation image mIoU: **60.000000%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
학습 체크포인트와 조합 선택에 valid를 사용하므로 독립 테스트 점수가 아닙니다.
원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model349_train_route_v1/outputs/checks/export_spq0gyay`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A0_synthetic | 60.0000% | [image](valid_mask_comparison/A0_synthetic.png) |
| A1_synthetic | 60.0000% | [image](valid_mask_comparison/A1_synthetic.png) |
| A2_synthetic | 60.0000% | [image](valid_mask_comparison/A2_synthetic.png) |
| A3_synthetic | 60.0000% | [image](valid_mask_comparison/A3_synthetic.png) |
| A4_synthetic | 60.0000% | [image](valid_mask_comparison/A4_synthetic.png) |
| A5_synthetic | 60.0000% | [image](valid_mask_comparison/A5_synthetic.png) |
| A6_synthetic | 60.0000% | [image](valid_mask_comparison/A6_synthetic.png) |
| A7_synthetic | 60.0000% | [image](valid_mask_comparison/A7_synthetic.png) |
| A8_synthetic | 60.0000% | [image](valid_mask_comparison/A8_synthetic.png) |
| A9_synthetic | 60.0000% | [image](valid_mask_comparison/A9_synthetic.png) |
| A10_synthetic | 60.0000% | [image](valid_mask_comparison/A10_synthetic.png) |
| A11_synthetic | 60.0000% | [image](valid_mask_comparison/A11_synthetic.png) |
| A12_synthetic | 60.0000% | [image](valid_mask_comparison/A12_synthetic.png) |
| A13_synthetic | 60.0000% | [image](valid_mask_comparison/A13_synthetic.png) |
| A14_synthetic | 60.0000% | [image](valid_mask_comparison/A14_synthetic.png) |
| A15_synthetic | 60.0000% | [image](valid_mask_comparison/A15_synthetic.png) |
| A16_synthetic | 60.0000% | [image](valid_mask_comparison/A16_synthetic.png) |
| A17_synthetic | 60.0000% | [image](valid_mask_comparison/A17_synthetic.png) |
| A18_synthetic | 60.0000% | [image](valid_mask_comparison/A18_synthetic.png) |
| A19_synthetic | 60.0000% | [image](valid_mask_comparison/A19_synthetic.png) |
