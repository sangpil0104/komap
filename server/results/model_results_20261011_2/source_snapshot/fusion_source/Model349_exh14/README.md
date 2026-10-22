# KoMaP Model349 exh14 (14가중치, 전수 조합 탐색판)

Model349의 23개 체크포인트 중 일부를 덜어내는 **모든 조합(8,093개)**을 같은 융합·후처리로 채점해, valid20에서 가장 높았던 14가중치 조합을 고정한 추론 패키지입니다. 학습은 하지 않으며, 가중치는 Model349 릴리스의 **FP32 원본 체크포인트를 변환 없이** 사용합니다.

이 ZIP에는 추론 코드·고정 설정·가중치 해시·조합 탐색 스크립트·실행 결과만 있습니다. **가중치와 대회 데이터는 포함하지 않습니다.** [checkpoint_manifest.json](checkpoint_manifest.json)의 14개 파일(Model349 릴리스 `weights/`와 동일)을 `weights/` 아래 같은 경로로 준비해야 합니다.

## 재학습 재현 설계 (`oof_design/`, 2026-10-09 추가)

위의 `infer_lite.py`는 **공개 가중치 14개**로 추론합니다. 같은 구성을 처음부터 다시 학습하면 가중치 편차 때문에 점수가 0.2~0.5%p 움직입니다. 팀원 환경 재학습판은 80.05%였습니다. 또 아래 fold 선택과 결합 비율은 공개 가중치를 보며 valid로 고른 값이라, 재학습판에는 맞지 않을 수 있습니다.

`oof_design/`은 누가 다시 학습해도 같은 절차로 같은 수준이 나오도록 만든 학습 코드입니다.

- 모든 구성 요소를 시편 단위 4-fold × 시드로 학습합니다. 결정적 시드와 cuDNN 결정 모드를 씁니다.
- 구성, 밝기 배정, 결합 비율, 후처리는 OOF(fold 밖 예측)에서 고정된 규칙으로 고릅니다.
- 시험에서는 모든 fold × 시드 모델을 평균합니다.
- 기본은 **A 방식**입니다. train 70장으로 학습·선택하고, valid 20장은 마지막에 한 번만 채점합니다.
- 결과는 세 가지로 나눠 보고합니다. OOF 개발 점수, 바깥 fold 추정치(선택 과정 포함), 시드별 점수의 평균·표준편차입니다.

자세한 내용은 [oof_design/README.md](oof_design/README.md)에 있습니다. 검증 상태는 [oof_design/SMOKE_RESULT.md](oof_design/SMOKE_RESULT.md)에 있으며, 현재는 소형 실행(smoke)으로 코드 경로를 검증하는 중입니다. 전체 규모의 학습 결과는 아직 없습니다.

## 고정 구성

| 경로 | 체크포인트 | 융합에서의 역할 |
|---|---:|---|
| B4 322 (gray) | fold 3·4 | Al 보정 (gamma 0.6) |
| B4 326 (Raw/Z/CDF) | fold 1·2 | Primary Si 1.0, Al₃Ni 0.7 |
| B4 329 (가로 4000) | fold 1·3·4 | Al₃Ni 0.5, Primary Si 1.0 |
| HRNet336 | 1 | Primary Si 0.25, Al·Eutectic Si 0.75 |
| HRNet347 (역빈도 CE 미세조정) | 1 | Al 0.25 |
| 원본 해상도 269 | 1 | Eutectic Si 0.7 |
| 밝기별 고해상도 기본 경로 (TENT 5단계) | 4 | restart51, model109, seed456(restart59), pairwise(restart61); EMA 밝기 구간은 restart51 |
| **합계** | **14** | 1,755,428,956 바이트 |

Model349 원형 대비 제거: B4 322 fold 1·2, 326 fold 3·4, 329 fold 2, ResNet34 branch 3개(model97/98/109 branch), 표현 모델 181, EMA(model123). 결합 비율(gamma)·사전확률·필터·Primary Si 정리 규칙은 Model349와 같고, 제거된 구성의 gamma만 0입니다. 경량판들과 같이 고해상도 경로만 D4 TTA(768 타일, 원본+밝기 표준화 입력)를 쓰고 B4/HRNet/269는 단일 시점입니다.

## 실행

```bash
conda env create -f environment.yaml && conda activate komap   # 또는 requirements.lock.txt
python verify_bundle.py                                         # 가중치 14개 해시 확인
python -B -u infer_lite.py --input <images_dir> --output <새_출력_폴더> --gpus 0      # 여러 GPU: --gpus 0,1
python -B -u infer_lite.py --input <valid/images> --gt <valid/masks> --output <새_출력_폴더> --gpus 0
```

- 출력: `predicted_labels/`(0~3 라벨 PNG), `predicted_masks/`(제출용 색 마스크), `completion.json`, `--gt` 사용 시 `metrics.json`.
- 시작 전에 지정 GPU가 비어 있는지(메모리 64MiB 이하, 사용률 0) 확인하고, 사용 중이면 중단합니다.
- 클래스 순서 Al₃Ni, Primary Si, Al, Eutectic Si / 팔레트는 `configs/model349_lite.yaml`.

## 확인된 결과 (2026-10-08, 학과 P100, 원본 이미지부터 재추론)

평가는 이미지마다 4개 클래스 IoU를 평균한 뒤 valid20을 동일 비중으로 평균합니다(해당 이미지에 없는 클래스 IoU는 0).

| 구성 | 가중치 | valid20 image mIoU |
|---|---:|---:|
| Model349 원형 | 23 | 80.0683465% |
| lite14 (EMA 제거) | 14 | 80.0650399% |
| lite12 base (326 fold 1·3) | 12 | 80.1255610% |
| **exh14 (이 패키지)** | **14** | **80.2312313%** |

- 클래스별 평균 IoU: Al₃Ni 77.154, Primary Si 87.165, Al 94.349, Eutectic Si 62.257 ([results/valid20_metrics.json](results/valid20_metrics.json)).
- 실행 시간: valid20 260.5초(GPU 2개), test10 286.4초(GPU 1개), 단회 측정.
- 오프라인 재조합 점수(80.2310%)와 원본부터 재추론 점수가 일치했고, test10 마스크는 오프라인 재조합과 9,074,964픽셀 중 34픽셀만 달랐습니다.
- test10(정답 없음) 마스크 일치율: Model349 원형과 99.408%(53,745픽셀 차이), lite12 base와 99.621%. 일치율은 정확도가 아닙니다. 경량판 공통의 TTA 축소만으로도 원형과 약 0.6% 달라지며, exh14는 비교한 경량 후보 중 원형과 가장 가깝습니다([results/](results/)).

## 선택 방법과 한계

- 조합은 (322/326/329 fold 부분집합, HRNet336·347, 269, branch, pairwise, 고해상도 후보 경로)로 정의하고, 1단계 B4 fold 조합 4,096개, 2단계 나머지 구성 2,048개, 1·2단계 상위 교차 2,000개(중복 제외 8,093개)를 valid20에서 채점했습니다([search/](search/); 1단계는 서버 `exhaustive.py`, 2·3단계는 같은 확률을 float16으로 내보내 `exhaustive_local.py`로 계산). 저장 확률의 pairwise 혼합 비율이 0이어서 별도 pairwise 예측 on/off는 결과에 영향이 없었고, 이 패키지는 그 예측을 계산하지 않습니다.
- **valid20으로 고른 조합입니다.** Model349의 gamma·경로·후처리 역시 같은 valid20으로 정했으므로 원형도 같은 선택 편향을 가집니다.
- 시편(specimen) 하나를 빼고 나머지로 조합을 다시 고르는 nested 검증에서, 전수 탐색 절차는 lite14 대비 **+0.09%p [−0.12, +0.39]**(시편 bootstrap 95%)로 손해 근거는 없으나 개선이 확정되지도 않았습니다. 같은 방식에서 "326 fold 부분집합만 고르기"는 −0.14%p였습니다.
- lite14 대비 valid 이득 +0.166%p 중 상당 부분이 Primary Si가 없는 A71 한 장(+2.32%p, Al₃Ni·Eutectic Si 개선)에서 나왔습니다. A71을 빼면 이득은 +0.05%p이며, 그 조건에서도 이 조합은 8,093개 중 16위입니다.
- 이미지별 최악 변화는 lite14 대비 −0.28%p(A181)입니다.
- 따라서 test에서의 기대 이득은 원형 대비 작게(+0.05%p 안팎) 보는 것이 현실적입니다.

## 파일

- `infer_lite.py`: lite12 base 추론 코드에 322·347 경로와 B4 fold 선택 일반화를 추가.
- `configs/model349_lite.yaml`: 고정 구성(fold, gamma, 활성 경로).
- `checkpoint_manifest.json`, `recipe_manifest.json`, `verify_bundle.py`: 가중치·코드 해시 검증.
- `search/`: 전수 조합 채점(`exhaustive.py`), 분석·nested 검증(`analyze_exh.py`, `exh_summary.json`), test 마스크 비교(`test_combos.py`, `compare_test.py`).
- `results/`: valid20 점수·완료 기록, test10 완료 기록과 마스크 일치율.
