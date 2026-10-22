# 경계 보조 손실 복원 결과

train70만동일부모/seed81336/기존augmentation·LR·BN통계고정/4epoch 조건으로추가학습하고최종epoch4만원본valid20전체재추론했다. 다른13가중치/융합/라우팅/TENT5는유지했다. 20PNG/GT독립재채점은서버metrics의80 IoU와정확히일치했다.

| 구성 | valid image mIoU |
|---|---:|
| 원래base14 | 80.2312313% |
| 동일stream CE/Dice control | 80.1856429% |
| boundary 복원 | 80.1662518% |

원래base 대비-0.0649795%p, 같은CE/Dice control 대비-0.0193911%p. 최악A74_step3_sample4 -1.0055761%p. boundary손실제외를하락원인으로단정할수없고복원으로도개선이없었다.

| 클래스 | 원래base 대비 (%p) | control 대비 (%p) |
|---|---:|---:|
| Al3Ni | +0.2146577 | -0.0648666 |
| Primary Si | -0.1231827 | -0.0290892 |
| Al | +0.0744721 | -0.0082124 |
| Eutectic Si | -0.4258651 | +0.0246037 |

| 이미지 | boundary − base (%p) | boundary − control (%p) |
|---|---:|---:|
| A71_step2_sample1 | +0.6031345 | +0.0056590 |
| A11_step1_sample4 | +0.3203936 | -0.0370212 |
| A153_step3_sample3 | +0.1489387 | -0.0259114 |
| A63_step1_sample4 | +0.0779394 | -0.0590227 |
| A11_step2_sample4 | +0.0581597 | -0.0825682 |
| A170_step4_sample3 | +0.0559185 | +0.0332426 |
| A125_step1_sample4 | +0.0279778 | +0.0003267 |
| A55_step2_sample1 | +0.0208517 | +0.0070670 |
| A7_step1_sample4 | +0.0111239 | +0.0167649 |
| A59_step3_sample1 | -0.0490500 | -0.0645025 |
| A42_step2_sample4 | -0.0543307 | -0.1127646 |
| A117_step4_sample2 | -0.0773621 | -0.0121214 |
| A44_step4_sample3 | -0.0791045 | -0.0674015 |
| A191_step4_sample3 | -0.0871113 | +0.0061854 |
| A72_step4_sample4 | -0.1281865 | -0.2115010 |
| A67_step1_sample1 | -0.2178637 | -0.1211868 |
| A40_step3_sample3 | -0.2278259 | +0.0182003 |
| A181_step1_sample1 | -0.2851799 | +0.1046061 |
| A186_step2_sample2 | -0.4124371 | +0.0995781 |
| A74_step3_sample4 | -1.0055761 | +0.1145485 |

420optimizerupdates/학습924.58초,전체valid 단회451.50초. 동일train140파일/부모SHA/seed/segmentationloss·dataset·paletteSHA를검증했다. 원본source/weights불변receipt와종료PID를확인했다.

채택하지않고원래base유지. 반복사용valid의추가개발평가/사후가설이다. 독립일반화/81%달성/test점수를주장하지않는다.
