# 전역 공간 후처리100개 탐색

기존baseline20마스크/IoU정확재현 및최고후보원래predict함수20IoU정확일치확인. 저장확률재생이며원본전체재추론성적/독립일반화증거가아니다.

| 구성 | 캐시 image mIoU |
|---|---:|
| base14 | 80.2309769% |
| Eutectic edge blend0.125 | 80.2354351% |

PrimarySi gaussian_blend0.5/cleanup128은유지된다. 평균+0.0044582%p,최악A71−0.0819971%p. 조건부group bootstrap95%[−0.0150950,+0.0253183]%p,고정 grid시편제외선택−0.0225282%p. 매우작은상승이므로채택/추가raw실행하지않고탐색YAML만보존한다. 같은valid반복선택의한계를유지한다.

| 이미지 | 변화 (%p) |
|---|---:|
| A74_step3_sample4 | +0.1011573 |
| A170_step4_sample3 | +0.0597579 |
| A153_step3_sample3 | +0.0532629 |
| A40_step3_sample3 | +0.0498268 |
| A117_step4_sample2 | +0.0466109 |
| A7_step1_sample4 | +0.0326916 |
| A186_step2_sample2 | +0.0147363 |
| A59_step3_sample1 | +0.0113842 |
| A72_step4_sample4 | +0.0110650 |
| A55_step2_sample1 | +0.0004967 |
| A42_step2_sample4 | -0.0003394 |
| A63_step1_sample4 | -0.0005803 |
| A125_step1_sample4 | -0.0041744 |
| A44_step4_sample3 | -0.0059112 |
| A181_step1_sample1 | -0.0109441 |
| A11_step2_sample4 | -0.0129960 |
| A67_step1_sample1 | -0.0418925 |
| A191_step4_sample3 | -0.0579044 |
| A11_step1_sample4 | -0.0750869 |
| A71_step2_sample1 | -0.0819971 |
