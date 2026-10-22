# 전체 이미지 Soft IoU 보정 학습

앞선 patch CE/Dice 잔차, OOF 이식, 같은 source 평가, 밝기 정규화·제거 후보가 모두 기각된 뒤 정의한다. 공식 평가의 이미지별 4클래스 IoU 평균에 학습 단위를 직접 맞추려는 탐색이다. 개선을 보장하지 않으며 81% 목표는 미달이다. 반복 valid 개발평가와 독립 일반화 미확인의 한계를 유지한다.

## 고정 두 입력

첫 profile은 현재 base14의 train70 precleanup logits·raw gray·GT를 학습하고 같은 base14 검증 fields에 적용한다. 이 train 예측은 부모의 학습 이미지 예측이므로 독립 OOF가 아니다.

두 번째 profile은 감사와 raw 점수 재현을 통과한 Model397 network group OOF70 log-prob·raw gray·GT를 학습하고 같은 source 계열 Model401 확률에 적용한다. 외부 데이터와 valid/test 정답을 학습에 쓰지 않는다.

## 전체 이미지 목적함수

CNN 11,876params, zero-init, delta 상한 ±1, seed81349, AdamW LR1e-3/wd1e-4, cosine8epochs, FP32, 엄격한 결정성, gradient clip1을 유지한다. crop/positive sampling을 사용하지 않고 native 이미지 전체를 학습한다. 크기를 변경하지 않는다. batch1, 70images×8epochs=560optimizer updates, epoch별 seed된 이미지 순서와 HVflip/rot90를 사용한다. 최종 epoch8만 평가한다. epoch/seed/가중치를 valid로 선택하지 않는다. gray 입력은 이전 raw 기본을 유지한다.

각 이미지의 softmax p와 onehot GT y에서 클래스별 I=sum(p*y), U=sum(p)+sum(y)-I, J=I/max(U,1e-6)를 계산하고 4개 J의 평균을 사용한다. absent GT의 I=0이므로 J=0이다. hard onehot 예측에서는 공식 IoU와 일치하는지 빈 클래스 포함 합성 검사를 한다.

loss=.25CE+.75(1-mean4J)+.05mean(delta²). 이미지별로 동등한 가중치를 주며, pixel pooled 또는 patch batch Dice를 공식 평균으로 대체하지 않는다. Soft IoU 개선이 hard argmax 점수 개선을 보장하지 않으므로 실제 PNG로 검증한다.

## 실행과 검증

학습 loop cap20분, 유한 손실·gradient, 원본 shape, 70unique IDs, feature SHA, 원본 source/weights SHA, 초기 출력 일치를 검사한다. 두 profile을 한 GPU에서 순차 실행한다. 유휴 GPU와 기존 lock을 확인하고 다른 사용자 작업·기존 릴리스를 보존한다. Trainer는 train70만 읽고 evaluator에서만 valid20 GT를 읽는다.

PNG/GT 독립 공식 채점, 이미지·클래스 차이, 최악 손실, 시간, checkpoint 크기를 기록한다. 부모 출력 재사용은 개발 평가이고 전체 fresh 시간을 추정하지 않는다. 81%가 관측되면 전체 raw fresh 재현, 원본 불변, test10 마스크·일치율(정답 없음/test mIoU 없음), 한국어 재현 자료를 확인한 뒤 완료를 판정한다.

기존 base pointer, 원본 가중치, 릴리스, dirty tree를 보존한다. 현재 문서는 아직 구현·학습·새 valid 성적을 주장하지 않는다.
