# OOF 밝기 입력 고정 비교 결과

확률을 재사용한 개발 평가다. 전체 fresh 실행 시간이 아니며 독립 일반화 증거가 아니다. train70만학습/valid20PNG와GT독립공식채점/emptyunion0 확인.

| 구성 | image mIoU | 같은 source 대비 | 최악 이미지 |
|---|---:|---:|---|
| Model401 raw control | 79.4181709% | — | — |
| accuracy81_oof_zgray_20261009 | 79.2195997% | -0.1985712%p | A74_step3_sample4: -1.3196890%p |
| accuracy81_oof_nogray_20261009 | 79.1731641% | -0.2450068%p | A74_step3_sample4: -1.4655659%p |

둘다기각. 마지막epoch8만평가했고각8epochs/1120updates,checkpoint/sourceSHA,zero-refiner20maskexact,원본확률과14parent불변을확인했다. 밝기정규화/제거가해결책이라는증거없음. 기존base/릴리스유지/test mIoU없음/81미달.

## accuracy81_oof_zgray_20261009

| 이미지 | 같은 source 대비 |
|---|---:|
| A74_step3_sample4 | -1.3196890%p |
| A153_step3_sample3 | -0.9567546%p |
| A71_step2_sample1 | -0.8255904%p |
| A11_step1_sample4 | -0.5253067%p |
| A191_step4_sample3 | -0.4090901%p |
| A42_step2_sample4 | -0.3386989%p |
| A186_step2_sample2 | -0.3301033%p |
| A7_step1_sample4 | -0.2356263%p |
| A181_step1_sample1 | -0.2249099%p |
| A170_step4_sample3 | -0.1380007%p |
| A40_step3_sample3 | -0.0533866%p |
| A55_step2_sample1 | -0.0093739%p |
| A125_step1_sample4 | +0.0229207%p |
| A72_step4_sample4 | +0.0560279%p |
| A63_step1_sample4 | +0.1096596%p |
| A117_step4_sample2 | +0.1581248%p |
| A59_step3_sample1 | +0.1645078%p |
| A44_step4_sample3 | +0.2257760%p |
| A11_step2_sample4 | +0.3187552%p |
| A67_step1_sample1 | +0.3393339%p |

## accuracy81_oof_nogray_20261009

| 이미지 | 같은 source 대비 |
|---|---:|
| A74_step3_sample4 | -1.4655659%p |
| A153_step3_sample3 | -0.9867814%p |
| A71_step2_sample1 | -0.9434414%p |
| A11_step1_sample4 | -0.7594126%p |
| A42_step2_sample4 | -0.4960762%p |
| A191_step4_sample3 | -0.2614764%p |
| A7_step1_sample4 | -0.2463739%p |
| A181_step1_sample1 | -0.2068222%p |
| A125_step1_sample4 | -0.1686416%p |
| A63_step1_sample4 | -0.1211625%p |
| A40_step3_sample3 | -0.0988332%p |
| A55_step2_sample1 | -0.0985557%p |
| A170_step4_sample3 | -0.0283132%p |
| A72_step4_sample4 | +0.0096111%p |
| A59_step3_sample1 | +0.0713718%p |
| A44_step4_sample3 | +0.1012170%p |
| A186_step2_sample2 | +0.1519435%p |
| A117_step4_sample2 | +0.1802341%p |
| A11_step2_sample4 | +0.2209079%p |
| A67_step1_sample1 | +0.2460352%p |
