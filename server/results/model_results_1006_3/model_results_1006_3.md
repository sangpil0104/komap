# Model364 재현 실행 — 중단 결과 정리

**상태: 실패로 중단. 최종 Model364 검증 점수는 없습니다.**

- 실행 시작: 2026-10-06 16:57:24 (한국 시간)
- 종료: 2026-10-06 17:54:57 (한국 시간)
- 파이프라인 실행 시간: 0시간 57분 33초
- 완료: 26/57단계 (45.61%)
- GPU: 4,5,6,7 / conda: komap
- 원본 실행 경로: `/Data3/sppark/komap/experiments/model364_reproduction_gpu4567_v1/outputs/runs/model364_train_20261006T075724_689269Z`

## 완료된 valid20 분기 평가

| 분기 | image mIoU | 비고 |
|---|---:|---|
| Model336 | 78.622027% | 독립 분기 점수; 최종 Model364 점수 아님 |
| Model351 (D4) | 78.622542% | 독립 분기 점수; 최종 Model364 점수 아님 |

Model351 D4는 Model336 flip4 대비 +0.000515%p입니다. 이미지별 네 클래스 IoU를 평균한 후 20장 평균하며, 빈 union은 0입니다. global mIoU는 계산하지 않았습니다. 점수는 완료된 CSV 20행으로 재계산해 summary JSON과 일치를 확인했습니다.

## 중단 원인

`evaluate_model361`이 첫 이미지 처리 전에 다음 오류로 종료했습니다.

```text
ValueError: invalid or impractically dense tiling: patch_size=512, input_scale=3.0, native_tile=171, overlap=170, stride=1; use overlap below 90% of the native tile size
```

Model361의 12 epoch 학습과 `last.pt` 생성은 완료됐습니다. 검증에서는 scale=3 적용 후 원본 좌표 타일이 171픽셀인데 overlap을 170으로 지정해 stride가 1픽셀이 됐습니다. 추론 코드의 과도한 타일 중첩 방지 검사에서 중단됐습니다. 실행 준비 때 이 설정 조건을 검증하지 못한 문제입니다.

실패 당시 Model322 fold3 학습, Model257 학습, Model347 검증도 실행 중이었으며 실행기가 자식 작업들을 종료했습니다. Model347 검증 폴더에 완료된 summary/확률 archive가 없어 점수를 보고하지 않습니다. 기존 checkpoints와 실행 소스는 보존했습니다.

## 클래스별 이미지 평균 IoU

| Class | Model336 | Model351 D4 |
|---|---:|---:|
| Al | 94.375441% | 94.382902% |
| Al3Ni | 74.150212% | 74.211076% |
| Primary_Si | 86.125588% | 86.037031% |
| Eutectic_Si | 59.836866% | 59.859160% |

## 이미지별 image mIoU

| Image | Model336 | Model351 D4 |
|---|---:|---:|
| A117_step4_sample2 | 87.620343% | 87.688386% |
| A11_step1_sample4 | 71.929494% | 72.122734% |
| A11_step2_sample4 | 76.155990% | 76.646409% |
| A125_step1_sample4 | 79.806094% | 79.619772% |
| A153_step3_sample3 | 72.613252% | 72.574883% |
| A170_step4_sample3 | 86.127335% | 86.057003% |
| A181_step1_sample1 | 78.832644% | 78.682791% |
| A186_step2_sample2 | 79.302071% | 78.864853% |
| A191_step4_sample3 | 83.521763% | 83.503817% |
| A40_step3_sample3 | 82.833432% | 82.682194% |
| A42_step2_sample4 | 79.941334% | 80.127657% |
| A44_step4_sample3 | 89.265837% | 89.462998% |
| A55_step2_sample1 | 82.171203% | 82.264827% |
| A59_step3_sample1 | 82.220537% | 82.373753% |
| A63_step1_sample4 | 83.086020% | 82.920830% |
| A67_step1_sample1 | 74.723964% | 74.778290% |
| A71_step2_sample1 | 40.005202% | 40.033576% |
| A72_step4_sample4 | 85.928451% | 85.806078% |
| A74_step3_sample4 | 74.796179% | 74.748656% |
| A7_step1_sample4 | 81.559389% | 81.491334% |

## 작업별 상태

| 단계 | 상태 | 경과 시간 |
|---|---|---:|
| train_model6 | 완료 | 132.1초 |
| train_model62 | 완료 | 80.1초 |
| train_model97 | 완료 | 328.8초 |
| train_model98 | 완료 | 190.7초 |
| train_model109 | 완료 | 150.1초 |
| train_restart51 | 완료 | 192.7초 |
| train_restart59 | 완료 | 164.1초 |
| train_restart61 | 완료 | 333.3초 |
| train_model123 | 완료 | 136.1초 |
| train_model181 | 완료 | 198.6초 |
| infer_route47 | 완료 | 72.6초 |
| build_model183_cache | 완료 | 135.1초 |
| train_model194 | 완료 | 455.8초 |
| train_model195 | 완료 | 171.1초 |
| infer_model220_tent | 완료 | 437.8초 |
| train_model239 | 완료 | 298.2초 |
| train_model242 | 완료 | 342.8초 |
| train_model257 | 실행 중 종료 | — |
| train_model269 | 미실행 | — |
| train_model322_fold1 | 완료 | 2967.4초 |
| oof_model322_fold1 | 완료 | 26.5초 |
| train_model322_fold2 | 완료 | 2993.9초 |
| oof_model322_fold2 | 완료 | 34.0초 |
| train_model322_fold3 | 실행 중 종료 | — |
| oof_model322_fold3 | 미실행 | — |
| train_model322_fold4 | 미실행 | — |
| oof_model322_fold4 | 미실행 | — |
| aggregate_model322 | 미실행 | — |
| valid_model322 | 미실행 | — |
| train_model326_fold1 | 미실행 | — |
| oof_model326_fold1 | 미실행 | — |
| train_model326_fold2 | 미실행 | — |
| oof_model326_fold2 | 미실행 | — |
| train_model326_fold3 | 미실행 | — |
| oof_model326_fold3 | 미실행 | — |
| train_model326_fold4 | 미실행 | — |
| oof_model326_fold4 | 미실행 | — |
| aggregate_model326 | 미실행 | — |
| valid_model326 | 미실행 | — |
| train_model329_fold1 | 미실행 | — |
| oof_model329_fold1 | 미실행 | — |
| train_model329_fold2 | 미실행 | — |
| oof_model329_fold2 | 미실행 | — |
| train_model329_fold3 | 미실행 | — |
| oof_model329_fold3 | 미실행 | — |
| train_model329_fold4 | 미실행 | — |
| oof_model329_fold4 | 미실행 | — |
| aggregate_model329 | 미실행 | — |
| valid_model329 | 미실행 | — |
| train_model336 | 완료 | 2667.2초 |
| evaluate_model336 | 완료 | 102.1초 |
| train_model347 | 완료 | 425.8초 |
| evaluate_model347 | 실행 중 종료 | — |
| train_model361 | 완료 | 415.8초 |
| evaluate_model361 | 실패 | — |
| infer_model351_d4 | 완료 | 175.1초 |
| export_model364 | 미실행 | — |

## 보존 및 후속 작업

- 이 보고서는 새 결과 폴더에 저장했습니다. 기존 모델·소스·결과는 수정하지 않았습니다.
- 최종 예측이 없어 `valid_mask_comparison/`에는 설명 파일만 저장했습니다. 최종 비교 이미지 20장은 아직 생성할 수 없습니다.
- 재개하려면 별도 실험 경로에서 scale3의 타일 중첩 설정을 수정하고 완료된 체크포인트·작업을 재사용하는 실행기를 준비해야 합니다. 추론 overlap 변경은 원본 재현 설정과의 차이로 기록해야 합니다.
- 이번 정리에서는 학습이나 추가 추론을 시작하지 않았습니다.

[실행 상태](status.json) · [실패 로그](evaluate_model361.log) · [이벤트 기록](events.jsonl)
