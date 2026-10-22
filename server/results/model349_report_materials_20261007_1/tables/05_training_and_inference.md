# 학습·추론 설정

| 모델 | 패치 크기 | Batch | 누적 | 학습 길이 | Learning rate | Weight decay | Seed | 사용 checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model322 | 1536 | 2 | 1 | 16,000 steps/fold | 0.0003 | 0.01 | 0 | last.pt |
| Model326 | 1536 | 2 | 1 | 16,000 steps/fold | 0.0003 | 0.01 | 0 | last.pt |
| Model329 | 1536 | 2 | 1 | 16,000 steps/fold | 0.0003 | 0.01 | 0 | last.pt |
| Model336 | 512 | 4 | 2 | 70 epochs | 0.0003 | 0.0001 | 0 | last.pt |
| Model347 | 512 | 4 | 2 | 12 epochs | 0.0001 | 0.0001 | 347 | last.pt |
| Model269 | 1024 | 2 | 1 | 4 epochs (Model257 이후) | encoder 2.5e-6 / decoder 7.5e-6 / native 5e-5 | 1e-4 | 269 | best.pt |

모두 AdamW를 사용한다. B4는 300-step warm-up을 곱한 cosine schedule, HRNet은 epoch 단위 cosine schedule을 사용한다. B4의 모든 fold 인자에서 seed=0이지만 일부 worker seed는 시간에 의존하여 재학습의 완전한 결정성이 보장되지는 않는다.

| 계열 | 학습 손실 | 초기화 / 추가 설정 |
| --- | --- | --- |
| B4 322/326/329 | CE + Dice | ImageNet encoder; Lovasz/clDice/EMA 미사용 |
| HRNet336 | 0.5 CE + 0.5 Dice + boundary loss | ImageNet encoder; CE 클래스 가중치 없음 |
| HRNet347 | 0.5 weighted CE + 0.5 Dice + boundary loss | 336 last.pt에서 시작; encoder LR 배율0.3 |
| Model269 | weighted CE + Dice | Model257 best.pt에서 연속 fine-tuning |

Boundary loss = positive-reweighted BCE + Dice. Model347 CE 가중치는 팀 순서 [Al, Al3Ni, Primary Si, Eutectic Si]에서 [0.22906544, 0.49862612, 1.39166570, 1.88064275]이다.

| 분기 | 추론 크기/타일 | TTA | 확률 결합 |
| --- | --- | --- | --- |
| 기본 97/98/109 | 640/512/512, stride512/384/384 | 원본·수평·수직·90°회전의 4개 | uniform 타일 결합; 저장 캐시 fp16 |
| 고해상도·pairwise | 고해상도768/1024; pairwise768 | D4 8개 | Gaussian 타일 가중; 고해상도768:1024=0.8:0.2 |
| B4 322/326/329 | 전체 영상; 3배 또는 너비4000 | D4 8개 | 변환 복원 후 4-fold 평균 |
| HRNet336/347 | native256 → 입력512; overlap170, stride86 | 수평·수직 반전 조합4개 | Gaussian window로 타일 결합 |
| Model269 | 1024 타일, stride576, scale1 | D4 8개 | Gaussian 타일 결합 및 원본 크기 복원 |

기본 고해상도 분기의 TENT는 입력 영상마다 BN affine 파라미터를 entropy 최소화로 5 step 적응시키는 원래 추론 절차이다(lr=1e-5). 실제 test 출력은 기존 체크포인트와 고정 조합을 사용했으며 test 정답을 사용하지 않았다.
