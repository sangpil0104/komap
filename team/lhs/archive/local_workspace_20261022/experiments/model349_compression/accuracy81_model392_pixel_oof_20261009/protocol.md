# Model392 OOF 픽셀 보정 CPU 선별

목적은 새로운 valid 탐색 전에 이미 존재하는 Model392 seed1의 train70 그룹 OOF 확률에서 회색조 국소 질감과 확률을 결합한 비선형 보정이 유망한지 낮은 비용으로 확인하는 것이다. 이 실험은 Model349_exh14의 점수가 아니며 81% 달성 판정에 쓰지 않는다.

- Model392 seed1의 fold1~3 OOF 이미지/정답으로 LightGBM 4-class 픽셀 분류기를 학습한다. fold4의 17장만 고정 보류 평가한다. 입력은 원본 이미지의 회색조, 5·15픽셀 국소 평균/표준편차, Sobel gradient, 네 OOF 확률, 최대 확률과 top2 margin이다. GT는 feature 계산에 사용하지 않는다.
- 학습 이미지는 각 7,000개 균등 픽셀과 7,000개 저마진 픽셀을 seed39281로 추출한다. 동일 픽셀 중복은 허용하지 않는다. num_leaves15, 150 trees, learning_rate0.05, min_child_samples100, 2 CPU threads. 모델 성적을 보고 이 설정을 바꾸지 않는다.
- fold4에서 원래 Model392 확률과 보정기 확률을 0.25/0.5/1.0로 섞는 세 고정 후보를 평가한다. 네 클래스·이미지별 IoU의 empty union=0 평균을 사용한다. 기존 OOF 확률의 class order `[Al, Al3Ni, Primary Si, Eutectic Si]`를 유지한다. 양성 신호 기준은 가장 좋은 고정 비율의 17장 평균 변화가 +0.50%p 이상이며 최소 12장 개선, 최악 이미지 -0.50%p 이상이다. 통과해도 전체 4-fold paired OOF와 원본영상 valid 재추론 전에는 채택하지 않는다.
- 기존 OOF를 생성한 fold1~3 모델의 학습에는 fold4 정답이 포함된다. 보정기 자체에는 fold4 정답을 넣지 않지만, 따라서 이 보류 점수는 엄격한 독립 일반화 추정치가 아니다. valid/test 정답과 외부 자료는 열지 않는다. GPU와 원본 가중치/릴리스는 변경하지 않는다.
