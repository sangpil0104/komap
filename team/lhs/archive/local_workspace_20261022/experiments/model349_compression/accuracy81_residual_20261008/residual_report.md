# 작은 잔차 보정 모델 원본 valid20 검증

빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다.

| 구성 | image mIoU | base 대비 | 최악이미지손실 | 단회시간 |
|---|---:|---:|---:|---:|
| base14 | 80.2312313% | — | — | 별도환경단회 |
| refined_valid20 | 79.7361955% | -0.4950358%p | -5.3884568%p (A71_step2_sample1) | 521.20s |
| teacher_valid20 | 80.2312280% | -0.0000033%p | -0.0006271%p (A11_step1_sample4) | 450.79s |

원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는전체fresh부모valid+feature추출+잔차평가이며학습시간을제외한다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음.

## 채택 판단과 이미지·클래스 변화

후보는 기각한다. 같은 fresh parent 대비 −0.4950325%p이며,81% 목표를 달성하지 못했다. 초기 zero-refiner 20장 exact 검사를 통과해 입력 추출 오류와 모델의 실제 변화는 구분했다. 원본 source/14가중치는 변경되지 않았고 base pointer도 그대로다.

| 클래스 | 같은 fresh parent 대비 |
|---|---:|
| Al₃Ni | -0.8257924%p |
| Primary Si | -0.2980392%p |
| Al | +0.0460634%p |
| Eutectic Si | -0.9023617%p |

| 이미지 | 같은 fresh parent 대비 |
|---|---:|
| A71_step2_sample1 | -5.3888960%p |
| A11_step1_sample4 | -2.7264516%p |
| A181_step1_sample1 | -2.0951691%p |
| A170_step4_sample3 | -1.9673694%p |
| A153_step3_sample3 | -0.4691147%p |
| A186_step2_sample2 | -0.2590176%p |
| A42_step2_sample4 | -0.1891966%p |
| A191_step4_sample3 | -0.1549757%p |
| A63_step1_sample4 | -0.1527508%p |
| A55_step2_sample1 | -0.1142068%p |
| A59_step3_sample1 | -0.0192953%p |
| A11_step2_sample4 | +0.0339788%p |
| A44_step4_sample3 | +0.1054766%p |
| A125_step1_sample4 | +0.1305250%p |
| A72_step4_sample4 | +0.2057721%p |
| A67_step1_sample1 | +0.2867357%p |
| A40_step3_sample3 | +0.2983728%p |
| A117_step4_sample2 | +0.4319325%p |
| A7_step1_sample4 | +0.8790396%p |
| A74_step3_sample4 | +1.2639612%p |

9장 개선/11장 악화. 총15 checkpoint 1,755,478,682bytes; 보정 모델49,726bytes. 전체fresh valid pipeline521.2025초,같은parent450.7893초(모두단회). 속도개선근거가 아니다. test10은 이 기각 후보에 실행하지 않았으며 test 성능을 주장하지 않는다.

Al 이외 클래스가 평균적으로 악화됐다. 클래스 불균형이나 train 예측의 in-sample 성격은 가능한 원인이나, 이 결과만으로 원인을 확정하지 않는다. train70에서 추정한 클래스 비율을 사용한 고정 loss 후속 실험은 별도 후보이며 같은 valid 반복 선택의 한계를 계속 기록한다.
