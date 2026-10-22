# Model336 — HRNet_best recipe 재현 및 image-mean 평가

원본 구현: 비공개 [`CNUKoMaP/HRNet_best`](https://github.com/CNUKoMaP/HRNet_best), commit `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`.
관련 설계 참고: [`CNUKoMaP/sppark`](https://github.com/CNUKoMaP/sppark), commit `e0d71b1`.

이 격리 사본은 팀 HRNet-W48 + SMP U-Net++ 모델, 입력 2배의 512px patch recipe, 희귀상 중심 sampling, class-unweighted CE+Dice, 학습 때만 쓰는 class-agnostic 경계 보조 head, overlap-tile/flip 추론을 보존한다. 학습 entrypoint에는 P100에서 micro-batch 4 / effective batch 8을 쓰기 위한 FP16 autocast와 gradient accumulation만 더했다. 중단 뒤 재개할 수 있도록 epoch 단위 optimizer/RNG state를 inference checkpoint와 분리 저장한다.

추론·평가 기본값은 기존 4-way 가로/세로 flip TTA다. 선택 옵션 `--tta-mode d4`는 90도 회전과 반사를 포함한 8개 square-symmetry view의 확률을 원본 좌표계로 되돌려 평균한다. Model351은 checkpoint를 바꾸지 않고 valid-only 비교에만 이 옵션을 사용했다.

중요한 평가 차이: upstream `evaluate.py`는 pooled/global mIoU를 출력한다. 이 archive의 `evaluate_image_mean.py`는 empty-union=`0`, skip, one 세 정책으로 공식 image-mean 4-class mIoU를 출력하고 pooled mIoU는 진단 참고값으로만 표시한다. checkpoint와 확률맵은 매번 새 Model336 output 디렉터리에 저장하며 이전 실험을 덮어쓰지 않는다.

검토한 `sppark`에는 DINOv3 전역 문맥 + 국소 세부 CNN 결합 구조와 HRNet feature에 stride-2/stride-1 local detail을 더하는 설계 제안이 있다. archive의 `accuracy_v1` CNN baseline은 image-mean mIoU `78.4729%`로 보고되어 있으나, HRNet 설계 부분은 측정된 HRNet 결과가 아니다. 먼저 공개 HRNet-W48 recipe 전체를 통제된 branch로 시험했다. 이후 확장 아이디어는 `sppark` 방식의 full-image context/local-detail fusion과 Model270 class-wise 결합을 저장된 validation 확률로 평가하는 것이다.

이후 queue는 Model336 valid/test 추론 후 CPU-only 625-route 탐색(Model337)을 실행하도록 작성됐다. 각 상은 Model270 확률 유지 또는 HRNet 확률을 gamma `0.25/0.50/0.75/1.00`으로 혼합한다. route evaluator는 점수 계산 전 image 순서, GT mask 정렬, class order, 확률 정규화를 검사하며 raw valid 결과와 함께 leave-one-specimen-out route 선택도 보고한다.

**과거 실행 설정:** 당시 Model326→328→329→333 GPU queue가 끝난 뒤 P100에서 순차 실행하도록 했다. 서버 data root는 `/workspace/komap_model1/data`였다. 이 일정은 과거 queue 설명이며 현재 GPU 상태를 뜻하지 않는다.
