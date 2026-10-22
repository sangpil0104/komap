# Model364 — local image mIoU

Validation image mIoU: **95.687500%** (20 images)

각 이미지의 4개 클래스 IoU 평균 후 20장 평균. union=0이면 IoU=0. global mIoU는 계산하지 않습니다.
일부 선행 체크포인트 선택과 원본 조합 개발에 valid를 사용했습니다. 이번 최종 조합은 기록된 고정 설정이며 독립 테스트 점수가 아닙니다.
원본 대비 중간 모델 선택도 image mIoU(empty=0)로 통일했습니다. 원본 기록 점수는 참고값이며 새 모델의 점수가 아닙니다.
실행 및 소스 스냅샷: `/Data3/sppark/komap/experiments/model364_reproduction_gpu4567_v1/outputs/checks/synthetic_export_l7y8o30j`

비교: 좌상 Original / 우상 Error / 좌하 Prediction / 우하 Ground truth.
Error는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 색으로 표시합니다. 100%−image mIoU는 픽셀 오류율이 아닙니다.

| Image | image mIoU | Comparison |
|---|---:|---|
| A00_fixture | 88.7500% | [image](valid_mask_comparison/A00_fixture.png) |
| A01_fixture | 25.0000% | [image](valid_mask_comparison/A01_fixture.png) |
| A02_fixture | 100.0000% | [image](valid_mask_comparison/A02_fixture.png) |
| A03_fixture | 100.0000% | [image](valid_mask_comparison/A03_fixture.png) |
| A04_fixture | 100.0000% | [image](valid_mask_comparison/A04_fixture.png) |
| A05_fixture | 100.0000% | [image](valid_mask_comparison/A05_fixture.png) |
| A06_fixture | 100.0000% | [image](valid_mask_comparison/A06_fixture.png) |
| A07_fixture | 100.0000% | [image](valid_mask_comparison/A07_fixture.png) |
| A08_fixture | 100.0000% | [image](valid_mask_comparison/A08_fixture.png) |
| A09_fixture | 100.0000% | [image](valid_mask_comparison/A09_fixture.png) |
| A10_fixture | 100.0000% | [image](valid_mask_comparison/A10_fixture.png) |
| A11_fixture | 100.0000% | [image](valid_mask_comparison/A11_fixture.png) |
| A12_fixture | 100.0000% | [image](valid_mask_comparison/A12_fixture.png) |
| A13_fixture | 100.0000% | [image](valid_mask_comparison/A13_fixture.png) |
| A14_fixture | 100.0000% | [image](valid_mask_comparison/A14_fixture.png) |
| A15_fixture | 100.0000% | [image](valid_mask_comparison/A15_fixture.png) |
| A16_fixture | 100.0000% | [image](valid_mask_comparison/A16_fixture.png) |
| A17_fixture | 100.0000% | [image](valid_mask_comparison/A17_fixture.png) |
| A18_fixture | 100.0000% | [image](valid_mask_comparison/A18_fixture.png) |
| A19_fixture | 100.0000% | [image](valid_mask_comparison/A19_fixture.png) |
