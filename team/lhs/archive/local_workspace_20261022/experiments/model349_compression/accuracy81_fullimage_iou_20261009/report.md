# 전체 이미지 Soft IoU 학습 결과

부모 출력 재사용 평가이며 전체 fresh 실행이 아니다. train70만 학습하고 emptyunion0 이미지별4클래스평균/20평균을PNG/GT독립채점했다. source snapshot/SHA/학습receipt/native shape/합성hardIoU/zero-refiner20maskexact/원본불변이통과했다.

| 학습 입력 | 원본 mIoU | 보정 mIoU | 같은 원본 대비 | 최악 이미지 |
|---|---:|---:|---:|---|
| accuracy81_fullimage_parent14_20261009 | 80.2312280% | 79.6194302% | -0.6117978%p | A71_step2_sample1: -5.9226730%p |
| accuracy81_fullimage_oof397_20261009 | 79.4181709% | 79.0873207% | -0.3308501%p | A74_step3_sample4: -1.2026909%p |

둘다기각. 각8epochs/560updates,최종epoch8만평가,각checkpoint49,726bytes. native이미지SoftIoU로학습단위를맞추는것만으로개선되지않았다. 악화원인을확정하지않는다. 기존base/릴리스유지/test mIoU없음/81미달/독립일반화미확인.

## accuracy81_fullimage_parent14_20261009

| 클래스 | 같은 원본 대비 |
|---|---:|
| Al₃Ni | -1.3097247%p |
| Primary Si | -0.3396680%p |
| Al | -0.0663794%p |
| Eutectic Si | -0.7314190%p |

| 이미지 | 같은 원본 대비 |
|---|---:|
| A71_step2_sample1 | -5.9226730%p |
| A11_step1_sample4 | -3.3357898%p |
| A170_step4_sample3 | -1.8648700%p |
| A181_step1_sample1 | -0.6027666%p |
| A11_step2_sample4 | -0.5745028%p |
| A186_step2_sample2 | -0.5699953%p |
| A67_step1_sample1 | -0.3832657%p |
| A191_step4_sample3 | -0.3705421%p |
| A63_step1_sample4 | -0.3205011%p |
| A55_step2_sample1 | -0.2982929%p |
| A153_step3_sample3 | -0.2211741%p |
| A59_step3_sample1 | -0.1139054%p |
| A44_step4_sample3 | -0.0740194%p |
| A72_step4_sample4 | -0.0500490%p |
| A125_step1_sample4 | -0.0459930%p |
| A42_step2_sample4 | +0.2204757%p |
| A40_step3_sample3 | +0.2659506%p |
| A117_step4_sample2 | +0.3581707%p |
| A7_step1_sample4 | +0.5678062%p |
| A74_step3_sample4 | +1.0999810%p |

## accuracy81_fullimage_oof397_20261009

| 클래스 | 같은 원본 대비 |
|---|---:|
| Al₃Ni | -0.7757809%p |
| Primary Si | +0.0081302%p |
| Al | -0.2179202%p |
| Eutectic Si | -0.3378297%p |

| 이미지 | 같은 원본 대비 |
|---|---:|
| A74_step3_sample4 | -1.2026909%p |
| A153_step3_sample3 | -1.1369382%p |
| A71_step2_sample1 | -1.0276507%p |
| A186_step2_sample2 | -0.8290757%p |
| A11_step1_sample4 | -0.8236301%p |
| A191_step4_sample3 | -0.5322164%p |
| A7_step1_sample4 | -0.5213685%p |
| A63_step1_sample4 | -0.4062629%p |
| A67_step1_sample1 | -0.3934924%p |
| A59_step3_sample1 | -0.2345780%p |
| A55_step2_sample1 | -0.2338172%p |
| A170_step4_sample3 | -0.1299466%p |
| A125_step1_sample4 | -0.0836207%p |
| A11_step2_sample4 | -0.0695039%p |
| A72_step4_sample4 | -0.0633962%p |
| A40_step3_sample3 | -0.0153818%p |
| A44_step4_sample3 | +0.0255049%p |
| A117_step4_sample2 | +0.1510935%p |
| A181_step1_sample1 | +0.4164735%p |
| A42_step2_sample4 | +0.4934956%p |
