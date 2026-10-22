# 정확도 개선 실험 — 기존 모델과 소스 보존

새 실험은 이 폴더 안에서만 실행합니다. 기존 `komap/src`, `configs`, `runs`, `submission`, `README.md`와 모델 가중치는 수정하지 않았습니다. 실험 엔진은 `baseline/source/src/alphase`에 보관한 **기존 소스의 동일한 사본**을 읽습니다. 기존 editable 설치와 conda 환경도 바꾸지 않았습니다.

## 바로 실행

```bash
cd /Data3/sppark/komap
conda activate komap
bash experiments/accuracy_v1/run_suite.sh 8
```

이 명령은 아래 CNN 실험 4개를 순서대로 학습하고, 각 모델의 기본/D4 검증, 기존 모델을 포함한 seed 앙상블의 검증·테스트 출력, 비교표 생성을 수행합니다. **DINOv3는 기본 실행에 포함되지 않습니다.** GPU 1장 실행은 마지막 숫자를 `1`로 바꾸세요. 2장·4장도 지원합니다. 사용할 GPU는 `CUDA_VISIBLE_DEVICES`로 선택할 수 있습니다.

| 실험 | 기존 설정과 다른 점 | 총 updates |
|---|---|---:|
| `fullframe` | 512 패치 2,000 + 768 패치 1,000 + 전체 영상 1,500 | 4,500 |
| `boundary` | 기존 2,500/1,500/500 일정 + 경계 보조 손실 0.05 | 4,500 |
| `seed2027` | 기존 일정·손실 유지, seed만 2027 | 4,500 |
| `seed2028` | 기존 일정·손실 유지, seed만 2028 | 4,500 |

각 실험은 새로 초기화해 학습합니다. 기존 `best.pt`에 이어 학습하는 방식이 아닙니다. 모델 클래스, 학습 데이터 70장, 검증 20장, effective batch 8, 학습률 등은 유지하고 위 표의 변경만 적용합니다. `fullframe`과 `boundary`를 한 모델에 동시에 적용하지 않아 각 효과를 비교할 수 있습니다.

기존 측정 속도로 계산하면 GPU 8장 기준 전체 CNN 실험은 대략 **40~55분 규모**의 계획 예산입니다. 경계 손실 비용과 데이터 로딩 등에 따라 달라지며 새 실험 4개 전체를 실제로 학습해 측정한 시간은 아닙니다. 각 실행의 터미널과 웹 화면에서 ETA가 갱신됩니다. DINOv3 시간에는 이 추정을 적용할 수 없습니다.

중간에 멈춘 경우 같은 `run_suite.sh 8` 명령으로 다시 실행합니다. 완료된 실험은 설정·데이터 일치를 확인한 뒤 재사용하고, 중단된 학습은 같은 GPU 수에서 `last.pt`로 재개합니다. 서로 다른 설정의 기존 실험을 덮어쓰는 옵션은 제공하지 않습니다.

## 보존 구조

```text
experiments/accuracy_v1/
├── baseline/                     # 별도 복사한 기존 자료, 파일은 읽기 전용
│   ├── source/                   # 기존 src/configs/scripts/tests/docs/환경/공식 DINO 소스
│   ├── run/                      # 기존 official run 전체: best/last/stage 가중치, 점수, 로그
│   ├── submission/               # 기존 테스트 예측
│   ├── data/                     # manifest, folds, palette, audit
│   └── artifacts/                # 기존 환경·검사·벤치마크 기록
├── configs/                      # 이번 실험의 설정
├── lab/                          # 새 실험 관리·캐시·확률 앙상블·비교 코드
├── tests/                        # 이번 코드의 검사
├── run.py                        # 공통 실행 진입점
├── run_suite.sh                  # 전체 CNN 실험 실행
├── preservation.json             # 기존 파일 838개의 바이트 수와 SHA-256
├── baseline_manifest.json        # 보관 사본 376개의 SHA-256
└── outputs/                      # 새 결과는 모두 이 안에 저장
    ├── runs/<실험>/official/
    ├── evaluations/<실험>/official/valid/<추론설정>/
    ├── ensembles/seed_ensemble/official/{valid,test}/
    ├── cache/                    # 모델·데이터·설정 hash별 float32 확률
    ├── reports/                  # comparison.html, 비교 CSV
    └── smoke*/                   # 짧은 동작 검사 자료, 본 실험과 분리
```

원본 dataset은 읽기만 하며 복사본 manifest에 기록된 기존 PNG를 사용합니다. 원본 데이터와 고정 사본의 변경을 검사할 수 있습니다:

```bash
python -B experiments/accuracy_v1/run.py verify
```

이 검사는 모든 기존 파일과 보관 사본의 해시를 비교합니다. 기존 파일을 사용자가 나중에 변경했다면 해당 경로를 알려주며, 그 변경을 자동으로 되돌리지는 않습니다.

## 개별 실험과 재개

```bash
python -B experiments/accuracy_v1/run.py list

python -B experiments/accuracy_v1/run.py train fullframe --gpus 8
python -B experiments/accuracy_v1/run.py train boundary --gpus 8
python -B experiments/accuracy_v1/run.py train seed2027 --gpus 8
python -B experiments/accuracy_v1/run.py train seed2028 --gpus 8

# 중단된 fullframe 실험 재개
python -B experiments/accuracy_v1/run.py train fullframe --gpus 8 --resume
```

최고 검증 체크포인트는 `outputs/runs/<실험>/official/best.pt`, 재개용은 `last.pt`입니다. 각 단계는 직전 단계의 최고 가중치로 시작합니다. 상태·loss·검증 기록·TensorBoard 기록도 같은 run 폴더에 저장됩니다.

`--set key=value`로 고급 설정을 바꿀 수 있지만, 기본 suite는 고정된 네 실험의 설정과 같은 run만 재사용합니다. 이미 학습을 시작한 실험의 설정을 바꿔서 resume할 수는 없습니다.

## 진행 화면

별도 터미널에서 전체 실험 진행 화면을 실행하세요:

```bash
conda activate komap
python -B experiments/accuracy_v1/run.py monitor suite --port 8765
```

http://localhost:8765 에서 전체 학습 진행률·남은 학습 시간 추정·현재 작업과 모델별 최고 점수를 확인할 수 있습니다. 실험 이름을 클릭하면 각 학습의 loss·검증 mIoU·예측 예시 화면이 열립니다. 결과 비교표도 같은 화면에서 열 수 있습니다. 원격 VS Code에서는 Ports 탭에서 8765를 전달하세요. `outputs/suite_status.json`에는 전체 실행의 완료 목록이 기록됩니다.

개별 실행만 모니터링하려면 `monitor suite` 대신 `monitor fullframe`처럼 실험 이름을 지정하세요. 전체 화면의 ETA는 학습 업데이트 기준이며 후속 TTA·앙상블 평가 시간은 별도입니다.

## 단일 모델의 검증·테스트 출력

```bash
# 기존 모델 사본으로 D4 검증: 기존 결과는 그대로 둠
python -B experiments/accuracy_v1/run.py evaluate baseline --d4

# 새 fullframe 모델: 기본 추론과 D4를 각각 비교
python -B experiments/accuracy_v1/run.py evaluate fullframe
python -B experiments/accuracy_v1/run.py evaluate fullframe --d4

# 새 모델로 테스트 10장 PNG 생성
python -B experiments/accuracy_v1/run.py evaluate fullframe --split test --d4
```

단일 모델의 테스트 PNG는 `outputs/evaluations/fullframe/official/test/d4_1/predictions/`처럼 저장됩니다. 기존 `submission/predictions`를 덮어쓰지 않습니다. 동일 모델·데이터·설정의 확률은 hash와 provenance가 일치할 때만 캐시에서 재사용합니다.

## seed 3개 확률 앙상블

`configs/ensemble_seeds.yaml`은 다음 세 모델의 **float32 클래스 확률을 같은 비율로 평균한 뒤 마지막에 argmax** 합니다.

1. `baseline/run/best.pt`: 기존 seed 2026 최고 모델
2. `outputs/runs/seed2027/official/best.pt`
3. `outputs/runs/seed2028/official/best.pt`

```bash
python -B experiments/accuracy_v1/run.py ensemble --split valid
python -B experiments/accuracy_v1/run.py ensemble --split test
```

기본은 각 모델에 D4를 적용합니다. 모델은 한 번에 하나씩 GPU에 올려 확률을 저장하므로 세 모델을 동시에 GPU에 유지하지 않습니다. 클래스 순서, 입력 hash, 공간 크기, 학습/검증 ID, 추론 설정이 일치해야 결합할 수 있습니다. 학습에 사용한 이미지를 검증으로 평가하는 경우와 동일 체크포인트를 중복하는 경우는 거부합니다. 테스트 PNG 10장은 색상·크기·ID 검사까지 수행합니다.

앙상블 설정을 바꿀 때는 새로운 `name`을 사용하세요. 기존 결과의 이름으로 다른 가중치나 멤버를 덮어쓰지 않습니다. 단일 실험에서 `fullframe`/`boundary`의 개선을 확인했다면 새로운 앙상블 설정에 넣어 추가 효과를 비교할 수 있습니다.

## 점수 비교

```bash
python -B experiments/accuracy_v1/run.py compare
```

`outputs/reports/comparison.html`을 브라우저로 열면 다음을 확인할 수 있습니다:

- 검증 mIoU와 기존 78.4729% 대비 변화(%p)
- 초정 Si / Al₃Ni / 공정 Si / Al의 IoU
- 20장 중 개선된 이미지 수
- 이미지별 차이 CSV와 원본 평가 CSV 링크

이 표는 동일한 공식 검증 20장의 **개발용 비교**입니다. 여러 실험에서 선택한 최고 점수를 별도의 최종 테스트 성능으로 해석하지 않습니다. smoke 결과는 비교표에서 자동 제외합니다.

## group CV와 최종 90장 학습

```bash
# 선택한 설정으로 5-fold 중 하나 실행; fold0~fold4 반복
python -B experiments/accuracy_v1/run.py train fullframe --split fold0 --gpus 8
python -B experiments/accuracy_v1/run.py evaluate fullframe --model-split fold0

# 설정과 학습 길이를 확정한 후 새로 90장 학습
python -B experiments/accuracy_v1/run.py train fullframe --split all90 --gpus 8
python -B experiments/accuracy_v1/run.py evaluate fullframe --model-split all90 --split test --d4
```

검증 명령은 해당 체크포인트에 저장된 held-out ID를 사용합니다. `all90`에는 독립 검증셋이 없으므로 `--split valid` 평가를 거부하고, 고정 학습 길이의 마지막 raw 가중치를 사용합니다. 기본 seed 앙상블에는 official baseline이 들어 있으므로 그 설정은 official split 전용입니다. nested CV/OOF 보정기 자동화는 이 실험 묶음에 포함하지 않습니다.

## DINOv3-L 실험

```bash
python -B experiments/accuracy_v1/run.py train dino_large --gpus 8 \
  --pretrained /absolute/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

원래 공식 파일명을 유지하세요. 가중치가 없으면 학습을 시작하지 않습니다. 기존 hybrid 모델 소스를 그대로 사용하고, 설정은 총 4,500 updates, backbone 고정 → 마지막 4 blocks → 마지막 8 blocks → 전체 영상(마지막 8 blocks)으로 구성했습니다. CNN과의 구조 비교 후보이며 학습 일정까지 동일한 실험은 아닙니다.

공식 가중치가 현재 없어 이 모델의 실제 사전학습 로드·GPU 메모리·성능은 아직 검증하지 못했습니다. 먼저 별도 smoke 실행으로 사용 환경에서 확인할 수 있습니다:

```bash
python -B experiments/accuracy_v1/run.py train dino_large --gpus 8 --smoke \
  --pretrained /absolute/path/to/original_official_filename.pth
```

## 동작 검사

```bash
python -B -m pytest -c experiments/accuracy_v1/pytest.ini -q experiments/accuracy_v1/tests
python -B experiments/accuracy_v1/run.py verify
```

이번 작업에서는 본 실험 4,500회 학습 대신 별도 `--smoke` 폴더에서 짧은 GPU 동작 검사만 수행합니다. 점검 결과는 `BUILD_REPORT.md`에 기록합니다.
