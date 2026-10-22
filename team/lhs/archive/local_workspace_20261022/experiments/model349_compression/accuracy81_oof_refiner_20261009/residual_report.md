# 작은 잔차 보정 모델 원본 valid20 검증

빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다.

| 구성 | image mIoU | base 대비 | 최악이미지손실 | 단회시간 |
|---|---:|---:|---:|---:|
| base14 | 80.2312313% | — | — | 별도환경단회 |
| refined_valid20 | 79.9993891% | -0.2318422%p | -2.2617428%p (A71_step2_sample1) | 10.14s |
| teacher_valid20 | 80.2312280% | -0.0000033%p | -0.0006271%p (A11_step1_sample4) | 450.79s |

원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는부모출력재사용후보정평가만이며전체fresh시간이아니다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음.

## OOF 학습 후보 판단

후보를 기각한다. PNG/GT 독립 공식채점79.9993891386%, 같은 부모80.2312280101% 대비−0.2318388715%p. 부모출력은재사용했으므로전체fresh새실행이아니다. 모델397의rawOOF학습입력과base14의배포logits차이가있으며,그차이가악화원인인지이결과만으로단정하지않는다. 원본14source/weightsSHA불변,zero-refiner20maskexact,8epochs/1120updates와train70정답만사용한학습을확인했다.

| 클래스 | 같은 원본 대비 |
|---|---:|
| Al₃Ni | -0.6918257%p |
| Primary Si | -0.1862216%p |
| Al | -0.0648791%p |
| Eutectic Si | +0.0155709%p |

| 이미지 | 같은 원본 대비 |
|---|---:|
| A71_step2_sample1 | -2.2621820%p |
| A11_step1_sample4 | -1.1341944%p |
| A181_step1_sample1 | -0.8773102%p |
| A11_step2_sample4 | -0.2888994%p |
| A191_step4_sample3 | -0.2664040%p |
| A186_step2_sample2 | -0.1957295%p |
| A55_step2_sample1 | -0.1923790%p |
| A125_step1_sample4 | -0.1821512%p |
| A72_step4_sample4 | -0.1625151%p |
| A170_step4_sample3 | -0.1076866%p |
| A63_step1_sample4 | -0.0875765%p |
| A153_step3_sample3 | -0.0853146%p |
| A40_step3_sample3 | -0.0727054%p |
| A44_step4_sample3 | -0.0552095%p |
| A59_step3_sample1 | -0.0506990%p |
| A117_step4_sample2 | +0.0680737%p |
| A42_step2_sample4 | +0.0826568%p |
| A7_step1_sample4 | +0.1942312%p |
| A67_step1_sample1 | +0.2580155%p |
| A74_step3_sample4 | +0.7812018%p |

Model397기존raw OOF79.9858620081%의엄격한재현을통과했다. 클래스순서변환시argmax동률218pixels의우선순위를원래team순서로보존했고main순서argmax79.9858657475%를별도기록했다. 기존base/릴리스는변경하지않았다.독립일반화/test mIoU/속도개선주장없음.81%목표미달.
