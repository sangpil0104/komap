**표 8. 모델별 입력·공간 전처리**

| 경로 | 입력 표현 | 크기/통계 | 적용 이유와 주의 |
| --- | --- | --- | --- |
| R 고해상도 | 원본 RGB + 밝기 표준화 | 768 tile, stride 576 | 학습 평균·분산과 원본의 style 혼합 |
| R B4 322 | grayscale 3채널 | 3배 확대, crop 1536 | ImageNet encoder 입력 형식 유지 |
| R B4 326 | raw/z-score/CDF | 3배 확대, 전체 영상 통계 | 상대 밝기·밝기 순위 표현 |
| R B4 329 / E B4 | raw/z-score/CDF | width 4000, crop 1536 | 영상 너비 기준 배율 통일 |
| R native 269 | 전체 영상 표현 통계 | scale 1, patch 1024 | 원본 해상도 경로 포함 |
| R HRNet | RGB 정규화 | scale 2, patch 512 | 고해상도 특징 경로 |
| E HRNet | RGB 정규화 | fold별 modal width 정규화 + scale 2 | 저장된 target_width metadata 사용 |
| F | E log probability + gray | full-image gray z-score, crop 256 | 추론도 같은 전체 영상 통계 사용 |

실제 실행 소스와 학습 args 기준이다. [S04–S06, S12]
