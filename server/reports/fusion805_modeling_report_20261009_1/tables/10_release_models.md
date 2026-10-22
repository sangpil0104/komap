**표 10. R 구성 모델과 학습 계보**

| 구성 | 개수 | 초기값/부모 | 학습량 | seed |
| --- | --- | --- | --- | --- |
| Model109 ASPP | 1 | 보존 Model98 | 4 epochs, patch512 | 42 |
| restart51 | 1 | 새 Model109 | 6 epochs, patch768 | 123 |
| restart59 | 1 | 새 Model109 | 5 epochs, patch768 | 456 |
| restart61 pairwise | 1 | 새 restart51 | 4 epochs, patch768 | 42 |
| Model269 native | 1 | 보존 Model257 | 4 epochs, patch1024 | 269 |
| B4 Model322 | 2 | ImageNet; folds3·4 | 각 16,000 iterations | 각 0 |
| B4 Model326 | 2 | ImageNet; folds1·2 | 각 16,000 iterations | 각 0 |
| B4 Model329 | 3 | ImageNet; folds1·3·4 | 각 16,000 iterations | 각 0 |
| HRNet336 | 1 | ImageNet | 70 epochs | 0 |
| HRNet347 | 1 | 새 HRNet336 | 12 epochs | 347 |
| 합계 | 14 | Model98·257 이전 계보는 범위 밖 | 각 구성 1회 재학습 | — |

기존 완성된 R 가중치를 이번 출력으로 재사용하지 않았다. 단, 보존된 선행 Model98·257과 ImageNet 사전학습 초기값은 사용했다. [S03, S12]
