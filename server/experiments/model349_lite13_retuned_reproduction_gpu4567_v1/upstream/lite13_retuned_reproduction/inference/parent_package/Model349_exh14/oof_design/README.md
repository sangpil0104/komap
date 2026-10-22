# exh14 OOF fold × 시드 재학습 설계 (`oof_design/`)

이 폴더는 exh14의 모든 구성 요소를 **fold × 시드**로 다시 학습하는 코드입니다. 이렇게 학습한 OOF(fold 밖) 예측으로 구성·밝기 배정·결합·후처리를 고정된 규칙에 따라 고릅니다.

목표는 특정 가중치에 맞춘 점수가 아닙니다. **누가 다시 학습해도 같은 절차로 같은 수준이 나오는 모델**입니다. 상위 폴더의 `infer_lite.py`(공개 가중치 14개로 추론)는 그대로 유지됩니다.

## 설계

1. **데이터 고정** (`prepare_data.py`)
   - 기본은 학습 70장만 씁니다(A 방식). `--valid`와 `--confirm-valid-training-allowed`를 함께 주면 검증 20장도 합칩니다(B 방식). 대회 규칙이 허용하는지는 실행자가 확인해야 합니다.
   - 같은 시편(`A<번호>_`)의 이미지는 모두 같은 fold에 넣습니다. 4-fold이고, 이미지 수가 균형이 맞도록 결정적으로 배정합니다.
   - 입력 파일의 SHA-256과 fold 배정을 `data_manifest.json`에 고정합니다.
2. **fold × 시드 학습** (`train_fold_seed.py`)
   - 기존 Model349 학습 레시피를 그대로 씁니다. 바꾸는 것은 fold별 데이터 폴더와 시드(기록된 시드 + 1000 × S)뿐입니다.
   - **OOF 이미지의 정답은 학습, 체크포인트 선택, 전처리 통계 어디에도 쓰지 않습니다.**
     - ResNet/ASPP 계열은 fold 학습 이미지 안에 따로 둔 시편 단위 선택용 분할(약 15%)로 `best.pt`를 고릅니다.
     - B4와 HRNet은 마지막 체크포인트를 씁니다.
     - 밝기 사전확률(`calibration.json`)과 HRNet347의 역빈도 클래스 가중치는 그 fold의 학습 이미지로만 계산합니다.
3. **OOF 예측** (`predict_oof.py`)
   - fold별 OOF 이미지를 시드마다 따로 예측해 저장합니다. 정답은 읽지 않습니다.
4. **설정 선택과 점수** (`select_oof.py`)
   - 시드를 평균한 OOF으로 고릅니다.
   - 탐색 공간과 순서는 `fusion.py`의 `GRID`·`ORDER`로 고정합니다(밝기 대역 8가지, 구성원 포함 여부, 상별 결합 비율, 후처리).
   - exh14 설정에서 출발해 좌표 상승을 최대 3회 반복합니다. 이득이 1e-6 이하이면 움직이지 않습니다.
5. **시험 추론** (`infer.py`)
   - 모든 fold × 시드 모델의 확률을 구성 요소별로 평균합니다. 모델은 하나씩 불러 누적합니다.
   - 4번에서 고른 설정을 적용합니다. 사전확률은 정답 있는 전체 이미지로 계산한 값을 씁니다.

## 실행

기본은 **A 방식**입니다. train 70장으로 학습하고 OOF70으로 설정을 고른 뒤, valid 20장은 맨 마지막에 한 번만 채점합니다. valid의 이미지와 정답은 학습·선택 단계에서 전혀 읽지 않습니다.

```bash
cd Model349_exh14/oof_design
python prepare_data.py --train DATA/train --work WORK                       # train 70장만
for k in 1 2 3 4; do for s in $(seq 0 99); do python train_fold_seed.py --work WORK --fold $k --seed $s; done; done
for k in 1 2 3 4; do python predict_oof.py --work WORK --fold $k; done
python select_oof.py --work WORK --workers 32                                # 설정 선택 (OOF70)
python infer.py --work WORK --images DATA/valid/images --gt DATA/valid/masks --output OUT_VALID   # valid 1회 채점
python infer.py --work WORK --images TEST/images --output OUT_TEST         # 제출 마스크
```

B 방식(90장)은 첫 줄을 `--valid DATA/valid --confirm-valid-training-allowed`로 바꾸면 됩니다. 이 경우 valid가 학습에 들어가므로 valid 채점 줄은 빼고, 성능은 `select/report.json`의 바깥 fold 추정치로 봅니다.

### 권장: 측정용(A)과 제출용(B) 두 단계

| 단계 | 데이터 | 설정 선택 | 성능 측정 | 실행 차이 |
|---|---|---|---|---|
| A. 측정용 | train 70장 | train70 OOF | valid 20장을 한 번만 채점 | `prepare_data.py`에서 `--valid`를 빼고, 마지막에 `infer.py --images VALID/images --gt VALID/masks` |
| B. 제출용 | train + valid 90장 | OOF | `select/report.json`의 바깥 fold 추정치 | 위 명령 그대로. 대회 규칙이 허용할 때만 |

- A는 지금까지의 valid20 점수(exh14 80.23%, fusion805 80.47%)와 같은 20장으로 비교할 수 있습니다. 다만 레시피의 고정값 다수가 과거에 valid를 보며 정해졌다는 한계가 남습니다.
- B는 학습 데이터가 29% 늘고, 측정이 90장 기준이라 더 안정적입니다.
- GPU가 부족하면 A만 해도 되고, A의 모델도 그대로 제출에 쓸 수 있습니다.

- fold와 시드는 서로 독립이라 GPU 여러 개에 나눠 돌려도 됩니다.
- 끝난 단계는 `STAGE_DONE.json`으로 건너뜁니다. 미완성 폴더는 덮어쓰지 않습니다.

## 보고하는 점수 (`work/select/report.json`)

| 항목 | 의미 |
|---|---|
| `dev_oof` | 전체 OOF에서 설정을 고른 뒤 같은 이미지로 잰 점수. 최종 설정의 개발 점수 |
| `outer_estimate` | 바깥 fold j마다 나머지 fold의 OOF로 설정을 다시 고르고, fold j로 잰 점수를 합산한 값. **설정 선택까지 포함한 성능 추정치** |
| `per_seed.mean` / `per_seed.sd` | 최종 설정을 시드 하나의 OOF 예측에 적용한 OOF 점수의 평균과 표준편차. **재학습 재현성** |
| `default_exh14_config_oof` | exh14의 원래 설정을 그대로 적용한 OOF 점수 (비교용) |

`dev_oof`은 시드를 평균한 앙상블의 점수라서, 시드별 점수의 평균(`per_seed.mean`)과 다릅니다.

## 결정성과 재현 범위

- `runtime/sitecustomize.py`가 `PYTHONPATH` 맨 앞에서 실행됩니다. cuDNN autotuning을 끄고 결정적 커널을 우선 사용하며, 학습 코드가 나중에 `cudnn.benchmark = True`를 넣어도 무시합니다.
- 바뀐 학습 코드는 B4 322의 시각 시드 한 줄뿐입니다(`vendor/PATCHES.md`). B4 326/329는 기존 `--deterministic_workers` 옵션을 씁니다.
- 같은 GPU 종류·드라이버·라이브러리에서는 같은 결과를 목표로 합니다. **서로 다른 GPU 간의 비트 단위 일치는 보장하지 않습니다.** 환경 간 차이는 시드 평균으로 줄입니다.

## 그대로 이어받은 것

- 밝기 대역 경계(`choose_source`), 스타일 가중치(`route_params`), TENT 5단계(lr 1e-5), 타일 크기, Primary Si 정리 규칙은 과거 레시피에서 고정된 값입니다. 이번 OOF 선택 대상이 아닙니다.
- 기존 Model349 계보의 Model181·EMA·ResNet branch는 exh14에 쓰이지 않아 학습하지 않습니다. EMA 밝기 대역은 restart51로 처리합니다.
- fold 부분집합 선택은 하지 않습니다. 모든 fold × 시드를 평균합니다.

## 비용

P100 기준 fold × 시드 1회당 B4 3종이 각각 약 4.6시간, HRNet336 70에폭이 약 3시간이고, ResNet 계열 체인이 더해집니다. 시드 100개 × 4 fold이면 수천 GPU시간이 듭니다.

## 검증한 범위

`SMOKE_RESULT.md`를 참고하세요. 학습량을 최소로 줄인 실행으로 코드 경로만 확인했습니다. 전체 규모의 학습·점수는 아직 없습니다.
