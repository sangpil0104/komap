# Augmentation 비교 실험

`/Data3/sppark/komap/spec.md`의 원본 보존 지침에 따라 새 코드와 결과는 이 폴더에만 둡니다. 이전 모델·모델 소스·학습 결과·accuracy_v1 코드를 수정하지 않습니다. 기존 고정 사본 `../accuracy_v1/baseline/source`에서 모델·학습 엔진·평가 코드를 읽고, 새 실행 프로세스에서만 데이터 로더와 터미널 진행 표시를 연결합니다.

**진행 상태는 터미널 tqdm만 사용합니다. 웹 모니터, localhost 접속, HTML 대시보드는 사용하지 않습니다.** 기존 기록용 `status.json`, TensorBoard 이벤트 파일은 저장하지만 서버를 실행하지 않습니다.

## 기존에도 augmentation이 적용돼 있습니다

- 학습: D4(90도 회전·반전), 밝기 ±0.04, 대비/감마 0.9~1.1을 확률 0.5로 적용.
- 패치: 무작위 중심, 공정 Si 중심, 경계 중심의 혼합 추출.
- 마지막 전체 영상 단계: 회전·반전은 유지하고 밝기 변화는 끔.
- 추론 D4/TTA는 학습 augmentation과 별개입니다.

기존 설정의 `gaussian_noise_std`는 0이며, 원본 데이터 로더에는 노이즈 적용 코드가 없습니다. 기존 `augmentation.scales`도 `[1.0]`만 지원합니다. 새 노이즈/배율 변환은 이 실험의 별도 데이터 어댑터에서 구현했습니다.

## 실행

```bash
cd /Data3/sppark/komap
conda activate komap
bash experiments/augmentation_v1/run_suite.sh 8
```

세 가지 후보를 seed 2026으로 각각 새로 학습하고, 기본 추론과 D4를 평가한 뒤 터미널/CSV/TXT 비교표를 만듭니다. 기존 모델에 이어 학습하지 않습니다. 시작 시 기존 파일 2,094개의 해시를 검사합니다. 본 실험 전체는 아직 실행하지 않았으므로 정확도 개선은 보장하지 않습니다.

| 이름 | 기존 설정 대비 유일한 augmentation 변경 |
|---|---|
| `photo_mild` | 밝기 ±0.06, 대비/감마 0.85~1.15, 적용 확률 0.75 |
| `noise_mild` | 확률 0.25로 Gaussian noise 표준편차 0~0.01 추가(밝기 범위 0~1) |
| `scale_mild` | 확률 0.5로 배율 0.9/1.0/1.1 중 선택, 영상 bilinear·마스크 nearest 보간 |
| `control` | 기존 augmentation 그대로 재학습; 기본 suite에는 미포함 |

세 후보 모두 모델·손실·학습률·데이터 분할·4,500 updates(2,500/1,500/500)·effective batch 8을 유지합니다. **마지막 전체 영상 단계는 모두 기존과 동일**하며 노이즈·배율 변환을 끕니다. 배율 후보는 1.0도 선택하므로 실제 크기가 바뀌는 확률은 약 1/3입니다. 변환 후 실제 영상 전체에서 z-score/CDF 채널을 다시 계산하고 패치를 추출합니다. 마스크에는 밝기/노이즈를 적용하지 않습니다. 검증·테스트에 학습 augmentation을 적용하지 않습니다.

기존 8 GPU CNN 한 번이 약 9.1분이므로 세 학습의 단순 기준 합계는 약 27분입니다. 새 변환·평가 비용은 추가되므로 실제 소요 시간은 실행 중 표시되는 ETA를 보세요. 한 모델의 tqdm에는 updates, 경과 시간, 단계, loss, 최고 mIoU, 단계별 속도를 반영한 ETA가 나옵니다. 전체 suite tqdm은 완료된 실험 수와 평균 소요시간 기반 ETA를 표시하며 첫 실험 완료 전에는 전체 ETA를 알 수 없습니다. D4 평가는 별도 이미지 진행률로 표시됩니다.

같은 suite 명령을 다시 실행하면 완료된 학습/평가는 설정·소스·데이터가 일치할 때 재사용하고 중단된 학습은 `last.pt`로 재개합니다. 중단 전 GPU 수를 유지하세요. 평가 중 중단되어 불완전한 폴더가 남았다면 해당 새 평가 폴더를 다른 이름으로 보관한 뒤 재실행합니다. 자동 덮어쓰기는 하지 않습니다.

## 개별 실행과 반복

```bash
python -B experiments/augmentation_v1/run.py train photo_mild --gpus 8
python -B experiments/augmentation_v1/run.py train noise_mild --gpus 8
python -B experiments/augmentation_v1/run.py train scale_mild --gpus 8

# 중단된 학습 재개
python -B experiments/augmentation_v1/run.py train scale_mild --gpus 8 --resume

# 추가 seed에서 대조군과 후보를 함께 비교 (기본 실행 이후 필요할 때만)
python -B experiments/augmentation_v1/run.py suite --gpus 8 \
  --experiments control photo_mild --seeds 2027 2028

# 같은 group fold에서 비교
python -B experiments/augmentation_v1/run.py train control --split fold0 --gpus 8
python -B experiments/augmentation_v1/run.py train photo_mild --split fold0 --gpus 8
python -B experiments/augmentation_v1/run.py evaluate control --split fold0 --d4
python -B experiments/augmentation_v1/run.py evaluate photo_mild --split fold0 --d4
```

실제 동일 시편 ID인지 확인되지 않은 A-prefix 기준 fold임을 유지합니다. `compare`는 공식 검증 20장만 비교하며 fold 결과는 해당 평가 폴더에서 봅니다. `all90` 학습은 지원하지만 독립 검증셋이 없어 평가를 거부합니다. 이번 도구는 augmentation 학습/검증 비교에 집중하며 테스트 PNG 생성 명령은 추가하지 않았습니다.

## 점수 확인

```bash
python -B experiments/augmentation_v1/run.py evaluate photo_mild
python -B experiments/augmentation_v1/run.py evaluate photo_mild --d4
python -B experiments/augmentation_v1/run.py compare
```

- 모델: `outputs/runs/<이름>/seed2026/official/{best,last}.pt`
- 학습 기록: 같은 폴더의 `metrics.jsonl`, `status.json`
- 검증: `outputs/evaluations/<이름>/seed2026/official/{plain,d4}/metrics_summary.json`
- 비교: `outputs/reports/comparison.csv`, `comparison.txt`

변화량은 반드시 동일한 추론 방식의 기존 seed2026 점수와 비교합니다(plain 78.4729%, D4 78.6865%). 다른 seed는 대조군도 같은 seed로 실행해 확인하세요. CSV에는 빈 클래스 처리 3개 방식과 클래스별 집계 이미지 수, 모든 클래스가 존재하는 동일 이미지 집합에서의 변화도 기록합니다. 앞선 A71 사례처럼 빈 클래스 처리만으로 주 지표가 크게 변했는지 확인할 수 있습니다. 기존 평가 규칙은 변경하지 않습니다.

## 보존·동작 검사

```bash
python -B experiments/augmentation_v1/run.py verify
python -B -m pytest -c experiments/augmentation_v1/pytest.ini -q experiments/augmentation_v1/tests

# 각 단계 1 update, smoke 전용 경로: 본 실험과 분리
python -B experiments/augmentation_v1/run.py suite --gpus 8 --smoke
```

`preservation.json`은 이 작업 전 기존 파일 전체의 크기와 SHA-256을 보관합니다. 기존 파일을 사용자가 나중에 바꿨다면 검사가 실패하며 자동 복구하지 않습니다. `augmentation_source_sha256`을 체크포인트 설정에 넣어 새 데이터 어댑터가 변경된 상태에서 잘못 resume하지 않도록 합니다. 추가 패키지 설치는 필요 없습니다.

기하 변환은 영상/마스크에 동기 적용하고 마스크에는 nearest 보간을 사용한다는 원칙은 [Albumentations 공식 segmentation 문서](https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/)를 참고했습니다. 이 구현은 기존 NumPy/Pillow/PyTorch만 사용합니다. 위 강도와 확률은 현재 데이터에서 비교할 실험 가설이며 효과가 입증된 최적값이 아닙니다.
