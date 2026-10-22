# 이미지별 명암 임계값 모델

원본 이미지의 밝기 분포에서 **임계값 3개를 자동으로 찾고, 4개 밝기 구간을 4개 상으로 변환하는 CPU 모델**입니다. 신경망이나 사전학습 백본은 사용하지 않습니다. 기존 실험과 소스, 결과는 변경하지 않습니다.

## 동작 방식

1. 원본 해상도를 유지한 채 PIL의 `RGB.convert('L')`로 0–255 명암값을 구합니다. 투명 픽셀은 오류로 처리합니다.
2. 각 이미지의 히스토그램에 **4수준 Multi-Otsu**를 적용합니다. 클래스 간 명암 분산이 최대가 되도록 세 임계값 `T1 < T2 < T3`를 동적 계획법으로 찾습니다. 라이브러리 설치가 추가로 필요한 scikit-image 대신 NumPy로 구현했습니다.
3. 밝기 구간은 `[0,T1]`, `(T1,T2]`, `(T2,T3]`, `(T3,255]`입니다. 임계값은 입력 이미지마다 다시 계산합니다.
4. **train 70장으로만** 밝기 구간과 `Primary Si / Al3Ni / Eutectic Si / Al`의 대응을 결정합니다. 가능한 24개 일대일 대응을 비교해 train image mIoU가 가장 높은 하나를 저장합니다. 동률이면 클래스 ID의 사전순으로 선택합니다.
5. 추론에서는 원본 이미지와 저장된 `model.json`만 필요합니다. 정답 없이 명암 임계값을 계산하고 고정된 구간→상 대응으로 분류합니다.

명암값 종류가 4개 미만인 이미지에서는 4수준 분할이 불가능합니다. 이때 train의 정상 이미지에서 구한 세 임계값의 위치별 중앙값을 사용하고 `threshold_source=train_fallback`으로 기록합니다. 모든 train 이미지가 이 조건이면 모델 생성을 중단합니다. 따라서 상수 이미지에 존재하지 않는 네 상을 인위적으로 만들지 않습니다.

이 방식에서 학습은 **70장의 라벨을 이용한 구간→상 대응 보정**입니다. 신경망 학습이나 epoch 반복은 없습니다. 완전한 무지도 방법으로는 밝기 구간의 이름이 어느 상인지 결정할 수 없어서 이 단계만 정답을 사용합니다. 검증·테스트 정답으로 임계값, 구간 대응, 후처리를 조정하지 않습니다.

Multi-Otsu 설명: [scikit-image 공식 예제](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html). 이 구현은 같은 클래스 간 분산 목적함수를 사용하며, 빈 히스토그램 구간의 임계값은 인접 관측 명암값 사이의 중간값으로 정합니다.

## 실행

프로젝트 루트 `/Data3/sppark/komap` 기준입니다. 기본 Python은 기존 `komap` 환경이며 필요한 패키지가 준비되어 있습니다. 다른 환경에서는 `KOMAP_PYTHON=/path/to/python`으로 지정합니다. GPU는 사용하지 않습니다.

데이터 검사만 실행하려면 다음 명령을 사용합니다. 모델을 학습하거나 생성하지 않습니다. 원본 100장과 train/valid 정답 90장을 검사하며, test 10장은 정답 없이 검사합니다.

```bash
bash experiments/adaptive_threshold_v1/run_cpu.sh preflight
```

사용자가 실험을 시작할 때 다음 한 줄을 실행하면 **train 보정 → valid 20장 평가 → 결과 문서·사진 저장**을 수행합니다.

```bash
bash experiments/adaptive_threshold_v1/start_nohup.sh
```

```bash
tail -f experiments/adaptive_threshold_v1/logs/latest.log
```

표준 출력과 오류는 실행마다 새 로그 하나에 모입니다. 각 이미지 완료 시 진행률, 단계 경과 시간, 예상 남은 시간, 예상 종료 시각(시간대 포함)을 기록합니다. 첫 검증 이미지 처리 후에는 전체 경과 시간에 남은 검증·그림 생성 시간을 더한 총 예상 시간도 표시합니다. 마지막 요약 문서 저장 시간은 추정에서 제외되며, 종료 시 실제 총 경과 시간을 남깁니다. 시작 직후에는 측정값이 없으므로 임의의 완료 시간을 표시하지 않습니다. 기존 로그는 보존하고 `latest.log` 링크만 최신 로그로 바꿉니다. 같은 실행기로 중복 실행하면 잠금으로 막습니다.

포그라운드 실행 또는 단계별 실행:

```bash
bash experiments/adaptive_threshold_v1/run_cpu.sh suite
bash experiments/adaptive_threshold_v1/run_cpu.sh fit
bash experiments/adaptive_threshold_v1/run_cpu.sh evaluate --model /path/to/model.json
```

실제 새 이미지 추론(정답 불필요):

```bash
bash experiments/adaptive_threshold_v1/run_cpu.sh predict \
  --model /path/to/model.json \
  --input /path/to/original.png
```

`--input`에는 여러 이미지 또는 이미지 폴더를 지정할 수 있습니다. 원본과 마스크가 섞여 있는 폴더 대신 원본만 있는 폴더를 지정합니다. CLI 옵션은 `run_cpu.sh <명령> --help`로 확인할 수 있습니다.

## 저장 결과

| 경로 | 내용 |
|---|---|
| `experiments/adaptive_threshold_v1/runs/<실행ID>/model.json` | 구간→상 대응, 대체 임계값, train 점수·기록, 소스 해시 |
| 같은 폴더의 `mapping_candidates.json` | 24개 대응의 train image mIoU |
| 같은 폴더의 `train_manifest.csv`, `sources_at_creation/` | 학습 데이터 목록, 실행 소스 사본 |
| `results/model_results_MMDD_N/` | 해당 날짜의 기존 최대 번호 다음에 새로 생성한 검증 결과 |
| 결과 폴더의 동명 `.md` | 방법, 전체·상별 image 평균 점수, 이미지별 임계값, 그림 링크 |
| `summary.json`, `per_image.csv`, `per_image.json` | 세 빈 클래스 정책의 점수, 이미지별 임계값·혼동행렬 |
| `valid_mask_comparison/` | 검증 20장의 원본 크기 2×2 비교 사진 |
| `histograms/` | 이미지별 명암 분포와 세 임계값 시각화 |
| `masks/` | 기존 프로젝트 색상표를 사용한 RGBA 예측 마스크 |
| `provenance.json`, `verification.json`, `sources_at_creation/` | 데이터·모델·소스 출처, 저장 그림 픽셀 검사, 평가 소스 사본 |
| `experiments/adaptive_threshold_v1/predictions/<실행ID>/` | 새 이미지 추론 결과·임계값·모델 사본 |

비교 사진은 **좌상단 원본 / 우상단 오류 / 좌하단 예측 / 우하단 정답**입니다. 우상단은 정답과 일치하면 흰색 `(255,255,255)`, 틀리면 **그 위치의 정답 상 색**입니다. Eutectic Si의 회색 `(204,204,204)`은 흰색과 구분합니다. 그림을 저장한 뒤 네 패널을 다시 읽어 원본 배열과 픽셀 단위로 대조합니다.

## 지표

주 지표는 이미지별 `mean(IoU_PrimarySi, IoU_Al3Ni, IoU_EutecticSi, IoU_Al)`를 구한 뒤 valid 20장에서 평균한 **image mIoU (%)**입니다. 상별 IoU도 이미지 단위 값을 평균합니다. global mIoU는 계산·선택 기준으로 사용하지 않습니다.

빈 클래스는 **정답과 예측이 모두 없을 때만 IoU=1**로 처리합니다. 정답은 없지만 예측이 있으면 IoU=0입니다. 사용자에게 선택을 위임받은 정책으로, 정확히 부재를 예측한 경우를 인정합니다. 비교를 위해 IoU=0 처리 및 빈 클래스 제외 방식도 함께 저장합니다. 제외 방식은 이미지마다 존재하는 클래스만 평균하므로 항상 4개를 평균하는 주 지표와 구분합니다.

## 확인할 한계

같은 밝기를 갖는 서로 다른 상은 이 모델로 구분할 수 없습니다. 특히 Primary Si와 Eutectic Si의 명암이 겹치면 혼동이 남을 수 있습니다. 조명 변화, 노이즈, 상의 면적 비율 변화, 특정 상이 없는 이미지에도 취약할 수 있습니다. 4개 명암 구간이 반드시 실제 4개 상과 일치하지는 않습니다. 위치·모양·주변 문맥은 사용하지 않으며 평활화·형태학 처리도 하지 않습니다. **테두리 문제 해결을 보장하는 모델이 아니라, 원본 명암만으로 가능한 성능을 확인하는 기준 실험**입니다.

과거 기록의 Primary Si만 평가한 이진 분할 IoU와 이 모델의 4상 image mIoU는 직접 비교할 수 없습니다.

## 구현 검사

```bash
/home/sppark/miniconda3/envs/komap/bin/python -m unittest discover \
  -s experiments/adaptive_threshold_v1/tests -v
```

작은 히스토그램의 완전 탐색과 DP 최적값 비교, 명암 이동에 따른 적응, train 전용 대응 선택, 퇴화 이미지 대체, 이미지별 평균 지표, 원본 해상도 오류 패널, 합성 데이터의 CLI 실행을 검사합니다. 합성 데이터는 임시 폴더에 만들고 삭제합니다. 실제 데이터 보정·학습은 사용자가 위 실행 명령을 수행할 때만 시작합니다.
