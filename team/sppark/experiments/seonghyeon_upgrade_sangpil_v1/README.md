# EfficientNet-B4 개선 실험

`input/alphase_b4_codex_spec.md`를 구현한 독립 실험 폴더입니다. 기존 모델·소스·결과와 `input/`은 보존했습니다. **실제 데이터 학습은 아직 시작하지 않았습니다.**

## 시작하기

```bash
cd /Data3/sppark/komap/experiments/seonghyeon_upgrade_sangpil_v1

# GPU 0–5 배정과 12개 실행 설정만 확인: 학습하지 않음
./run.sh launch

# 사용자가 학습을 시작할 때 실행
./run.sh launch --execute
```

기본 실행은 **E03a / E03b / E04 / E05 × seed 0,1,2 = 12개**입니다. GPU 0–5에 한 작업씩 배정하고 빈 GPU에 다음 작업을 넣습니다. GPU 6·7은 사용하지 않습니다. 각 seed는 대응하는 기존 seed의 `last.pt`에서 독립적으로 시작합니다. 한 GPU의 batch2를 유지하며 DDP로 배치를 늘리지 않습니다.

각 작업의 tqdm 진행률·경과 시간·예상 남은 시간은 `outputs/logs/*.log`에 남습니다. 실행 터미널에도 전체 작업 진행률이 표시됩니다. 예: `tail -f outputs/logs/E04_seed0.log`. 서버나 웹 대시보드는 없습니다. 실행 환경은 기존 검증 환경인 `experiments/seonghyeon_seed10_v1/.venv/bin/python`을 사용합니다.

## 모델과 비교 설정

공통 기반은 **SMP U-Net + EfficientNet-B4**입니다. 입력은 그레이스케일을 3채널로 복제하고 기존 mean0.449/std0.226으로 정규화합니다.

새 모델은 원본 좌표의 detail ROI를1536으로 확대하고, 다음 경로를 필요에 따라 추가합니다.

- TextureStem: 원본 밝기 + 원본 sigma4/16의 상대 밝기, 16→32→64채널 GN/SiLU.
- ContextCNN: detail의2배 영역을768로 입력, 32→64→128→256채널. 원본 좌표와 공유 D4에 맞춰 detail에 정렬합니다.
- ResidualFusion: 기존 decoder feature와 선택한 새 경로에서 4클래스 보정 logits를 계산합니다. 마지막 층을0으로 초기화하므로 처음에는 기존 logits와 같습니다.

| 설정 | 내용 | 업데이트 |
|---|---|---:|
| E00 | 기존 seed32 full-frame/D4 기준선 | 추론만 |
| E02 | 새 native ROI·tile geometry, 새 분기 없음 | E02_scratch로16000 |
| E03a | 분기 없는 추가 학습 대조군 | 4000 |
| E03b | detail feature만 사용하는 잔차 head | 500+3500 |
| E04 | E03b + 상대 밝기 TextureStem | 500+3500 |
| E05 | E03b + 넓은 ContextCNN | 500+3500 |
| E06 | E03a + random/mixed/hard sampler | 4000 |
| E07 | E03a + 원본 해상도 NLL/Lovasz 손실 | 4000 |
| E08_ema | E03a + EMA | 4000 |
| E08_bn | E03a + 기존 BN 통계 고정 | 4000 |
| examples/enhanced_candidate | texture/context/sampler/native loss 조합 후보 | 500+3500 |

조합 후보는 성능을 검증한 최종 모델이 아닙니다. E09에 채택할 변경과 최종 모델은 단독 실험 결과로 결정해야 합니다. 새로운 ROI/tile 추론은 기존 width4000 full-frame 추론과 다른 설정입니다. 기존 학습의 면적 비례 샘플링과 새 학습의 이미지 균등 샘플링도 구분합니다.

잔차 모델의 첫500 update는 base/BN을 동결하고 새 모듈만lr3e-4로 학습합니다. 다음3500 update는 새 optimizer로 encoder1e-5, decoder/head3e-5, 새 모듈1e-4를 사용합니다. Plain 대조군은 동결 구간 없이 base 학습률로4000 update입니다.

주지표는 **원본 이미지별4클래스 mIoU를 구한 뒤 이미지별 동일 가중 평균**, union=0이면1입니다. 검증은 valid20 전체/D4를1000 update마다 수행합니다. 학습 후 선택 checkpoint의 md, per-image/per-class CSV, confusion matrix, 비교 그림을 자동으로 새 `results/model_results_날짜_번호/`에 만듭니다. 그림은 원본/오류/예측/정답2×2이며 오류 패널에 image mIoU와100%−image mIoU를 표시합니다.

## 실행 예시

```bash
# 설정·필수 경로 검사
./run.sh validate-config --config configs/experiments/E04.yaml

# 단일 seed. --execute를 빼면 계획만 출력
./run.sh train --config configs/experiments/E04.yaml --seed 0 --execute

# 같은 설정으로 중단 지점부터 재개
./run.sh train --config configs/experiments/E04.yaml --seed 0 \
  --resume outputs/E04_seed0/last.pt --execute

# 다른 단독 변경을 GPU 0–5에서 비교
./run.sh launch --configs configs/experiments/E06.yaml configs/experiments/E07.yaml \
  configs/experiments/E08_ema.yaml configs/experiments/E08_bn.yaml --seeds 0,1,2 --execute

# ImageNet부터 시작하는 geometry 대조군
./run.sh launch --configs configs/experiments/E02_scratch.yaml --seeds 0,1,2 --execute

# 실제 새 checkpoint의 재평가
./run.sh evaluate --run-dir outputs/E04_seed0 --checkpoint best

# 합성 입력 forward/backward: optimizer update 없음
./run.sh smoke --config configs/examples/enhanced_candidate.yaml --full-size --steps 2

# 실제 train 패치 overfit smoke도 사용자가 명시적으로 시작
./run.sh smoke --config configs/examples/enhanced_candidate.yaml \
  --real-train-patches --steps 40 --execute
```

`last.pt`에는 model/optimizer/scheduler/EMA/RNG/sampler/config/split hash가 들어 있습니다. `best_image_miou.pt`와 `best_ema_image_miou.pt`는 별도입니다. `resume`은 상태 복원이고 `training.initialize_from`은 새 학습의 초기값입니다. Indexed RNG와 저장한 hard score map으로 ROI 선택도 복원합니다. BF16에는 GradScaler를 사용하지 않으며 nonfinite loss/gradient는 optimizer update 전에 실패합니다.

Hard sampler는 처음500 update 동안 cache가 없으면 같은 이미지의 random ROI로 대체합니다. 이후 hard mode에서 선택한 **train 이미지**만 현재 모델로 추론해 score map을 만들고1000 update 후 갱신합니다. 이는 `in_sample_train_mining`이며 refiner 학습용 OOF로 쓰지 않습니다. 외부 train cache를 사용할 때에는 `sampling.hard_cache`에 manifest를 지정합니다. Valid/test는 sampler에 넣을 수 없습니다.

## 기존100개 모델의 확률 앙상블

```bash
./run.sh reproduce --image-id A71_step2_sample1
./run.sh cache-probs --split valid --members all         # dry-run
./run.sh cache-probs --split valid --members all --execute
./run.sh ensemble --config configs/ensemble_uniform.yaml --execute
```

GPU에100개 모델을 동시에 올리지 않습니다. 각 seed의 원본 좌표 FP32 확률을 캐시하고 CPU FP64로 스트리밍 평균합니다. Mask 투표나 logits 평균이 아닙니다. 고정 seed 순서 K=1/3/5/10/20/50/100 곡선, 분산, 투표 비율, Al3Ni→EutecticSi 진단을 만듭니다. 실제100-model sweep은 아직 실행하지 않았습니다. 서로 다른 모델 cache는 `ensemble --prediction-manifests <manifest1> <manifest2> ... --execute`로 동일 가중 결합할 수 있습니다. 같은 이미지 해시·원본 크기·클래스 순서가 필요합니다.

## strict OOF 보정기

```bash
./run.sh build-oof --config configs/oof_train70.yaml           # fold/6-GPU 계획
./run.sh build-oof --config configs/oof_train70.yaml --execute # 사용자가 teacher 학습 시작
./run.sh validate-oof --config configs/oof_train70.yaml
./run.sh train-refiner --config configs/refiner.yaml --execute
```

5개 fold teacher(각56장)와 deployment teacher(70장)를 같은 ImageNet 초기화·고정16000 update·single seed로 학습합니다. 각 fold holdout은 teacher의 학습·checkpoint 선택·BN 보정에 쓰지 않습니다. teacher 실제 checkpoint와 registry까지 검증합니다. 기존100개 모델 및 그 모델에서 warm-start한 가짜 OOF는 허용하지 않습니다. Refiner는 원본1채널+확률4채널+상대 밝기2채널+entropy1채널을 받아4클래스 점수를 보정합니다. 학습은 train70의 OOF, 선택은 valid20입니다. Outer-fold 성능 검증은 아직 하지 않았습니다. A-prefix는 실제 specimen group으로 확인된 정보가 아닙니다.

## 최종 PNG 및 검증

`configs/final.yaml`은 아직 개선 모델을 선정하지 않았으므로 **기존 seed32 기준선**을 가리킵니다. 이번 구현 검사에서 생성한 `outputs/submission/`의 test10 PNG도 seed32 기준선입니다. 실제 test mIoU는 정답이 없어 계산하지 않습니다.

```bash
./run.sh validate-submission --config configs/final.yaml
# 다른 확정 모델로 생성할 때: checkpoint/config와 새 output 경로를 함께 지정
./run.sh predict --config configs/final.yaml --split test \
  --set submission.output=outputs/submission_new --execute
```

PNG는 원본 H/W와4색을 보존하며 이름은 `image_id_mask.png`입니다. 공식 zip/제출명 규칙은 제공되지 않아 PNG 검증까지만 수행합니다.

## 검사 기록

- [구현 진행과 실제 실행 내역](reports/progress.md)
- [기존 코드/데이터 감사](reports/repo_audit.md)
- [명세와 구현 매핑](reports/implementation_plan.md)
- [검증 보고서](VALIDATION_REPORT.md)
- `reports/seed32_reproduction.json`: valid20 image mIoU80.2265456654%, 과거 점수와 부동소수점 오차 범위에서 일치.
- `reports/seed32_baseline/`: seed32 실제 확률 캐시에서 생성한20장 진단/비교 그림.
- `reports/train_augmentation/`: train-only 밝기 통계·증강 예시. 입력 정규화에 GT 클래스 통계를 사용하지 않습니다.

CPU 수학 계약과 통합 테스트:

```bash
PYTHONDONTWRITEBYTECODE=1 /Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python -m pytest tests -q
```
