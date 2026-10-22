# lite13 구성

R 7개 + E 5개 + F 1개 = 13개 실제 가중치. lite14에서 R322 fold3 하나를 제거했다.

| 경로 | 계열 / 역할 | 개수 | 재현용 식별 |
|---|---|---:|---|
| R | ResNet34 기반 기본 모델 | 2 | restart61, model269 |
| R | EfficientNet-B4, 전체 이미지 통계 입력 | 1 | 326 seed3456 fold2 plain, 322(Al source) 역할과 공유 |
| R | EfficientNet-B4, 폭 4000 정규화 | 3 | 329 fold1·3·4 |
| R | HRNet-W48 | 1 | 347, 336 역할과 공유 |
| E | EfficientNet-B4ws | 2 | fold1·3, 평균 후 E 50% |
| E | HRNet-W48, N3 입력 | 1 | seed414, 다음 HRNet과 평균 후 E 25% |
| E | HRNet-W48, 폭 정규화 | 1 | 392 seed1 fold4 |
| E | HRNet-W48, train70 전체 학습 | 1 | seed5 epoch70, 단독 E 25% |
| F | 잔차 보정기 | 1 | epoch8 |

R 설정 변경은 `inference.hr_alias`에 `'322': '326'` 추가와 active_components에서 322 제거뿐이다.
Al source 결합 비율(al_gamma 0.6)을 포함한 다른 모든 값은 lite14와 같다.
