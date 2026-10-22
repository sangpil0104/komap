# lite20_r70: R의 B4를 계열당 train70 전체 모델 1개로 (팀원 실행용, 미실행)

**아직 실행하지 않은 실험입니다. 점수는 없습니다.**

## 목적
- lite20의 R은 B4 세 계열에서 fold 일부를 valid20 점수로 골라 씁니다(322 fold 3, 326 fold 1·2, 329 fold 1·3·4).
- 같은 레시피의 fold 중 일부를 valid로 고르는 것은 선택 편향입니다. nested 검증에서 "326 fold 고르기"는 −0.14%p였습니다.
- 이 변형은 fold를 고르지 않습니다. 계열마다 **train70 전체로 한 번 학습한 모델 1개**를 씁니다.
  - 학습 레시피(코드, 입력, 배율, 1536 영역, lr, 시드 규칙)는 fold 학습과 같습니다.
  - 학습 데이터는 fold 학습 이미지(평균 52.5장)에서 train70 전체로 바뀝니다.
  - 반복 횟수는 데이터 비율만큼 늘려 **21,000회**로 합니다(16,000 × 70/52.5). 그래서 이미지 한 장을 보는 횟수가 fold 학습과 같습니다. 이 값은 결과를 보기 전에 정했고, valid로 고르지 않습니다(`ITERS` 환경 변수로 바꿀 수 있지만 바꾸지 않는 것을 권장).
  - valid·test는 학습에도 체크포인트 선택에도 쓰지 않습니다(`--no_valid_eval`, 마지막 가중치).
- 결합 비율, 밝기 규칙, E, F는 lite20 그대로 둡니다. 다시 고르는 것은 없습니다.

| 구성 | R 가중치 | 전체 가중치 | R의 B4 |
|---|---:|---:|---|
| lite20 | 11 | 20 | fold 6개 (valid로 선택) |
| **lite20_r70** | **8** | **17** | train70 모델 3개 (선택 없음) |

## 실행

```sh
# 공개 가중치 위에서 (B4 3개만 새로 학습)
bash run_r70.sh 0 /path/DATA /path/WEIGHTS/R /path/WEIGHTS/assets /path/runs/r70_seed0 0,1,2

# 시드 S 재학습 결과 위에서 (run_retrain.sh S ... 가 끝난 폴더와 짝지어 비교)
bash run_r70.sh S /path/DATA /path/runs/seedS/R/weights /path/runs/seedS/assets /path/runs/r70_seedS 0,1,2 --allow-retrained
```

- `DATA`: `train/{images,masks}`(70장), `valid/{images,masks}`(20장), `test/images`(선택).
- GPU 목록의 GPU마다 B4 하나씩 동시에 학습합니다. 시작할 때 비어 있어야 하고, 시작한 뒤에는 다른 사용자가 들어와도 멈추지 않습니다.
- 학습이 끝나면 **같은 나머지 가중치로 lite20_r70과 lite20을 둘 다 추론**합니다. 그래서 B4 구성만 다른 짝 비교가 됩니다. 추론은 첫 번째 GPU를 씁니다.
- 결과
  - `OUT/summary.txt`, `summary.json`: valid20 두 점수, 차이, 시편 bootstrap 95% 구간, 개선/악화 장수, 최악 이미지, test10 마스크 차이
  - `OUT/lite20_r70/{valid,test}/fusion/predicted_masks`: 마스크
- 끝난 단계는 건너뛰므로, 끊기면 같은 명령으로 이어서 돌립니다.
- 학습만 따로 돌리려면 `bash r70/train_r70.sh SEED DATA OUT_WEIGHTS 0,1,2`를 씁니다.

## 시간 (P100 추정)
- B4 학습 1개는 21,000회라 약 6시간입니다(fold 학습 16,000회 약 4.5시간 기준). GPU 3장이면 약 6시간, 1장이면 약 18시간입니다.
- 추론(두 구성 × valid·test)은 약 1시간입니다.
- 시드 재학습과 함께 돌리면 R의 B4 fold 학습(322·326)은 lite20 비교용으로만 필요합니다.

## 판단 기준 (제안, 결과를 보기 전에 정해 두기)
- lite20의 valid 점수에는 fold 선택 편향이 들어 있습니다. 따라서 lite20_r70이 valid에서 **약간 낮게 나오는 것은 예상 범위**입니다.
- 제안하는 기준: 시드 평균 최종 점수가 lite20보다 **0.05%p 넘게 낮지 않으면** lite20_r70을 씁니다. 가중치가 적고 선택이 없기 때문입니다.
- 시드가 하나뿐이면 결과는 참고로만 씁니다. 차이가 bootstrap 구간 안이면 우연 범위로 봅니다.

## 한계
- 21,000회는 장당 학습량을 fold 학습과 맞춘 값이지, 70장에 최적인지 확인한 값은 아닙니다. 여러 값을 돌려 valid로 고르면 다시 선택 편향이 생기므로 하나로 고정했습니다.
- fold 평균(2~3개)에서 모델 1개로 바뀌면 확률이 더 날카로워질 수 있습니다. 결합 비율은 fold 평균에 맞춰 정해진 값이라 그 영향이 함께 섞입니다.
- R의 ResNet/ASPP 계열(269, restart51, pairwise)의 best.pt는 원래 파이프라인처럼 valid로 고릅니다. 이 변형에서도 그대로입니다.
- 코드 변경: `Model349_fusion805/Model349_exh14/infer_lite.py`에 선택 설정 `inference.b4_full70`을 추가했습니다. 설정이 없으면 기존 구성과 똑같이 동작합니다. 설정 파일은 `Model349_fusion805_lite23_b4ws/selected_{R,config,assets}_lite20_r70.*`입니다.
- 새 B4 3개는 SHA 검사를 하지 않습니다. 나머지 14개는 공개 가중치로 실행할 때 SHA를 확인합니다.
