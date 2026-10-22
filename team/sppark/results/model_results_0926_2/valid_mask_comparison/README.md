# 비교 이미지 표시

- 좌상 Original image / 우상 오류 위치 / 좌하 predicted mask / 우하 ground-truth mask.
- 우측 상단 두 줄: image mIoU(%)와 100% − image mIoU(%). 소수점 넷째 자리까지 표시합니다.
- image mIoU는 해당 이미지의 4개 클래스 IoU 평균이며 정답·예측 양쪽에 없는 클래스는 IoU=1입니다.
- 오류 위치는 정답 색, 맞게 예측한 위치는 흰색입니다.
- 두 수치의 글자 크기는 28px, 패널 간격은 24px입니다. 이미지와 마스크는 원해상도 그대로입니다.
- 이번 변경은 사용자 요청에 따라 기존 비교 사진을 백업 없이 덮어썼습니다.
- [표시 수치 CSV](../layout_updates/miou_complement_20260927_1/display_metrics.csv), [검증 기록](../layout_updates/miou_complement_20260927_1/verification.json).
