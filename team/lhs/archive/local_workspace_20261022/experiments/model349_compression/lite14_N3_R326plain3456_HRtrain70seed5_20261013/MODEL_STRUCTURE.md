# lite14 구성

R 8개 + E 5개 + F 1개 = 14개 실제 가중치.

| 경로 | 계열 / 역할 | 개수 | 재현용 식별 |
|---|---|---:|---|
| R | ResNet34 기반 기본 모델 | 2 | restart61, model269 |
| R | EfficientNet-B4, 확대 입력 | 1 | 322 fold3 |
| R | EfficientNet-B4, 전체 이미지 통계 입력 | 1 | 326 seed3456 fold2 plain |
| R | EfficientNet-B4, 폭 4000 정규화 | 3 | 329 fold1·3·4 |
| R | HRNet-W48 | 1 | 347, 336 역할과 공유 |
| E | EfficientNet-B4ws | 2 | fold1·3, 평균 후 E 50% |
| E | HRNet-W48, N3 입력 | 1 | seed414, 다음 HRNet과 평균 후 E 25% |
| E | HRNet-W48, 폭 정규화 | 1 | 392 seed1 fold4 |
| E | HRNet-W48, train70 전체 학습 | 1 | 새 seed5 epoch70, 단독 E 25% |
| F | 잔차 보정기 | 1 | epoch8, E log 확률 + 원본 밝기 |

R 상별 결합 → E가 Al3Ni 또는 Al로 예측한 영역 덮어쓰기 → F가 Al로 예측한 영역 최종 보정. 나머지 영역에는 앞 단계 라벨을 유지합니다.
