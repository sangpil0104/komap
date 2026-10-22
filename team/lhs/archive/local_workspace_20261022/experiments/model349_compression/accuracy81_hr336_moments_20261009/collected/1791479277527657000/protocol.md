# HR336 이미지별 BN 통계 보간 추론 후보

학습된 잔차 보정 후보들이 개선되지 않은 뒤 정의한 후속 탐색이다. 기존 ResNet34 AdaBN 실험과 HR336 후보를 구분한다. 81% 목표는 미달이고 같은 valid 반복 개발평가의 한계를 유지한다.

## 고정 범위

원본14 가중치·구성·TENT5·배합 계수·다른 분기를 유지한다. 바꾸는 것은 HR336의 추론 중 running mean/variance뿐이다. convolution, affine gamma/beta, checkpoint 파일은 변경하지 않는다. 원본 checkpoint를 읽어 CPU에서 BN 구조와 유한 통계를 감사한다. 다른 사용자 작업과 기존 릴리스·dirty tree를 보존한다.

## 이미지별 통계

모델을 eval 상태로 유지한다. 해당 이미지의 원래 HR336 타일 forward에서 각 BN 입력을 forward pre-hook으로 읽어 채널별 sum/sumsq/count를 수집한다. 원본 BN 통계와 affine 값은 수집 과정에서 바꾸지 않는다. 원래 patch512/overlap170/scale2/batch2/identity TTA를 유지한다.

현재 이미지의 mean=sum/count, population variance=max(sumsq/count-mean²,0)를 계산한다. 각각 원본 buffer와 현재 추정치를 (1-alpha)*original+alpha*current로 보간한다. 고정 후보 alpha=0.25,1.0을 각 한 번 평가한다. 이를 두 모집단의 분산을 추정한 값이라고 부르지 않으며 buffer의 선형 보간으로 정의한다. BN의 eps는 그대로 유지한다.

각 이미지 시작과 끝에 모든 BN buffer를 원본으로 복원한다. 다른 이미지의 통계와 정답을 통계 추정에 섞지 않는다. gradient·optimizer·정답 학습은 없다. 통계 수집·추론 단계에서는 valid GT를 읽지 않고, 후보 전체 확률 생성 후 공식 점수 평가에서만 GT를 읽는다. 기존 TENT와 마찬가지로 현재 추론 이미지에만 의존하는 동작이다. test 입력은 이 개발 실험에 사용하지 않는다.

## 필수 검사

합성 BN hook 통계와 직접 계산한 mean/variance 일치, 수집 중 원본 buffer 불변, alpha0 출력 exact, affine/비BN state 불변, 이미지 사이 복원을 검사한다. 실제 원본 HR336 확률과 baseline replay를 확인한다. 기존 다른6분기의 SHA를 검사해 읽기 전용 재사용하며 기본 raw parent80.2312280101%의20장/80IoU를 정확히 재현해야 후보 점수를 사용한다. 미세한 CUDA 차이가 발생하면 숨기거나 허용오차를 늘리지 말고 별도로 조사한다.

이는 changed HR336 원본 영상 추론과 다른 분기 확률 재사용을 결합한 component replay이다. 전체 fresh 새 실행 시간으로 보고하지 않는다. 후보 PNG/GT를 독립 공식 채점하고 이미지·클래스 손실/최악 이미지/실제 changed 경로 시간/변경 없는 checkpoint 크기를 기록한다. 동일 조건 단회 시간만으로 속도 향상을 주장하지 않는다.

한 GPU에서 고정 후보를 순차 실행한다. 유휴 GPU·기존 lock·실제 프로세스를 검사하고 다른 사용자 점유시 대기한다. 충돌하면 우리 프로세스만 종료하고 부분 산출물을 보존한다. 81%가 관측되면 전체 raw fresh 재현·원본 불변·test10 마스크/일치율(정답 없음/test mIoU 없음)·한국어 재현 자료를 확인한 뒤 목표 완료를 판정한다.

현재 구조 감사와 프로토콜은 후보 성능을 입증하지 않으며 아직 GPU 실험 성적은 없다.
