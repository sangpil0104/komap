# 같은 입력에서의 OOF 잔차 평가

원본 확률을 재사용한 탐색이다. 전체fresh 추론/독립일반화 성적이 아니다. 공식 emptyunion0 이미지별4클래스평균/20평균을PNG/GT독립채점했고metrics와일치한다.

| 구성 | image mIoU |
|---|---:|
| Model401 raw control | 79.4181709% |
| 같은 OOF 잔차 적용 | 79.2448654% |

같은 source 대비 -0.1733055%p. 후보기각. 최악A74_step3_sample4 -1.3974390%p.

| 이미지 | 같은 source 대비 |
|---|---:|
| A74_step3_sample4 | -1.3974390%p |
| A153_step3_sample3 | -1.0019475%p |
| A71_step2_sample1 | -0.7259708%p |
| A11_step1_sample4 | -0.5476800%p |
| A7_step1_sample4 | -0.2620896%p |
| A191_step4_sample3 | -0.2016720%p |
| A40_step3_sample3 | -0.1660635%p |
| A42_step2_sample4 | -0.1273451%p |
| A181_step1_sample1 | -0.1204390%p |
| A170_step4_sample3 | -0.1080117%p |
| A55_step2_sample1 | -0.0460056%p |
| A72_step4_sample4 | -0.0399176%p |
| A125_step1_sample4 | -0.0063134%p |
| A186_step2_sample2 | +0.0739801%p |
| A44_step4_sample3 | +0.1120825%p |
| A59_step3_sample1 | +0.1585178%p |
| A63_step1_sample4 | +0.1719468%p |
| A117_step4_sample2 | +0.2034802%p |
| A11_step2_sample4 | +0.2634107%p |
| A67_step1_sample1 | +0.3013673%p |

zero-refiner20maskexact와원본확률/parent14sourceweights불변을확인했다. Model397 OOF 네트워크 계열을맞춰도개선되지않았으므로앞선base14전이손실을입력차이만으로설명할수없다. 원인을확정하지않는다. 새학습/test사용없음. 기존base/릴리스유지/81미달.
