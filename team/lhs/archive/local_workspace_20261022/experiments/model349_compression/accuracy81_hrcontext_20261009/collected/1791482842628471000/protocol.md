# HR336 학습 문맥 확대: 고정 4 epoch 후보

기존 control 미세조정은 학습 손실 0.16623→0.15486으로 감소했지만 공식 valid는 80.18564%로 원래 base보다 낮았다. BN 이미지 적응은 더 크게 악화됐다. 이번 후보는 정규화 통계·추론 해상도를 그대로 두고 학습에서 관찰하는 문맥만 확대한다. 아직 개선 또는 81% 달성을 입증하지 않는다.

초기화는 동일한 원본 HR336 FP32 checkpoint다. 제공 train70/GT만 사용하고 valid/test GT 또는 외부 데이터·새 사전학습 가중치는 학습에 사용하지 않는다. 데이터 증강·positive sampling0.5·seed81336·70×12 samples/epoch·4epochs·microbatch2×accum4·0.5CE+0.5Dice·encoderLR5e-6/decoder2e-5·AdamW1e-4·cosine4epochs·FP16AMP·gradient clipping1·BN eval·boundaryhead frozen을 기존 대조군과 유지한다.

유일한 계획된 학습 설정 변경은 patch512→1024, input_scale2 유지로 native crop256→512다. batch1로 바꾸면 기존 Dice의 배치 집계가 달라지므로 batch2를 유지한다. 넓어진 영역으로 crop/padding/증강 결과는 달라지며, 모든 픽셀 또는 증강이 대조군과 짝지어졌다고 주장하지 않는다. 작은 이미지 반사 padding은 기존 dataset 방식이다. 업데이트 수420와 학습 샘플 수3360은 유지하지만 픽셀 수와 계산량은 4배이므로 계산량이 같은 비교는 아니다.

유휴 GPU3 및 기존 lock을 확인한다. 합성1024 batch2의 AMP forward/backward/finite gradient/실제 optimizer step을 먼저 수행하고 peak memory/원본 checkpoint SHA 불변을 기록한다. OOM이나 외부 GPU 점유는 우리 생성 자식만 중지하고 실패를 보존한다. 메모리 검증을 통과하기 전 실제 학습은 시작하지 않는다. 최대3900초로 학습을 제한하고 부분 epoch checkpoint/state를 보존한다. 자동 재학습이나 valid를 보고 epoch/LR/crop을 선택하는 과정은 없다.

마지막 epoch4만 공식 valid20으로 평가한다. 기존14개 구성 중 HR336만 최종 checkpoint로 교체하고 원본 identity TTA/patch512/scale2/TENT5/배합/후처리를 유지한다. 원본 영상부터 전체14개 경로를 새 실행하고 원본/기존 control과 비교한다. 공식 metric은 per-image4IoU 평균 후20장평균이고 emptyunion0이다. PNG/GT 독립 채점, 클래스/이미지 변화·최악 손실·실제 단회 전체 추론 시간·가중치 bytes·원본 SHA를 검증한다.

기존 control은 같은 source와 frozen 학습 설정의 이전 실행이며 동시 실행 대조군은 아니다. 반복 valid에서 개발한 후보로 독립 일반화 보장은 없다. raw81%를 관측해도 전체 재현과 원본 불변·test10 마스크(점수 없음)·한국어 재현 자료를 확인한 뒤 목표 완료를 판단한다. 기존 base/릴리스는 검증 전 교체하지 않는다.
