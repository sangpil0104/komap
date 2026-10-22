# Model349 lite20 r70 — full-train70 B4 가중치 3개

기존 HR336 seed 3407~3414의 B4 교체 평가에서 공통으로 사용한 B4-322/326/329 가중치입니다.
각 B4의 학습 seed는 **0**, 학습 데이터는 **train70 전체**, 학습 길이는 **21,000 iterations**입니다.
각 ZIP은 마지막 checkpoint `last.pt`, 원본 `args.json`, 설명 및 SHA256을 포함합니다.
세 파일 모두 원본 완료 기록 및 8개 시드 평가 패키지의 가중치와 SHA256이 일치합니다.

| 모델 | 다운로드 파일 | 크기 |
|---|---|---:|
| B4-322 | [Model349_lite20_r70_B4_322_weights.zip](Model349_lite20_r70_B4_322_weights.zip) | 72.32 MiB |
| B4-326 | [Model349_lite20_r70_B4_326_weights.zip](Model349_lite20_r70_B4_326_weights.zip) | 72.32 MiB |
| B4-329 | [Model349_lite20_r70_B4_329_weights.zip](Model349_lite20_r70_B4_329_weights.zip) | 72.33 MiB |

Git LFS로 저장됩니다. GitHub 파일 페이지에서 Download raw file을 누르거나 `git lfs pull`로 받으세요.
ZIP 형식은 일반 Deflate입니다. 체크섬은 `Model349_lite20_r70_B4_SHA256SUMS.txt`에 있습니다.

| 모델 | 입력 | 정규화 | 크기 처리 | 샘플링 |
|---|---|---|---|---|
| B4-322 | gray3 | imagenet | scale 3 | area |
| B4-326 | zcdf3 | image_global | scale 3 | area |
| B4-329 | zcdf3 | image_global | target width 4000 | uniform |

이는 R 구성의 full70 B4 가중치 3개이며, 전체 R/E/F 앙상블과 HR336 가중치는 별도입니다.
8개 시드 평가의 최고 R+E+F image mIoU 79.899812%는 HR336 seed 3412와 나머지 구성 모델을 결합한 점수입니다.
개별 B4 단독 점수를 의미하지 않습니다. seed 탐색 결과는 valid20에서 선택한 개발용 결과입니다.
