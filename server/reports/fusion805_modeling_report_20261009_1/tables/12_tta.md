**표 12. 실제로 실행된 추론 증강**

| 경로 | 실제 실행 | 주의 |
| --- | --- | --- |
| R 고해상도 base | D4 + TENT5, 768 tile | 원본/스타일 영상 결합 |
| R B4 | tta=False | YAML의 b4_tta 이름과 구분 |
| R HRNet | tta=False | tta_mode 문자열이 있어도 비활성 |
| R native269 | tta_mode=none | 학습 중 선택 검증의 D4와 다름 |
| E B4 | D4 | 4-fold family 평균 |
| E HRNet | flip4 | seed별 4-fold 평균 + width normalization |
| F | 단일 전체 이미지 잔차 예측 | F가 Al인 위치만 stage2에 반영 |

문서화되지 않은 추론을 추정하지 않고 infer_lite.py의 실제 호출 인자를 기준으로 기록했다. [S04, S05]
