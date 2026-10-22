# Model390 사전등록: per-item crop·augmentation RNG 교정

## 목적과 가설

Model382와 과거 학습 recipe의 DataLoader는 `worker_init_fn=seed_worker`, `persistent_workers=False`를 사용한다. P100의 고정 Albumentations 2.0.8 환경과 실제 `KomapPatchDataset` 경로의 합성 시험에서 legacy 코드는 worker마다 같은 private crop RNG 및 Compose RNG 초기 상태를 재사용했다. 별도 합성 시험에서는 `Compose.set_random_seed(s)`가 모든 child transform에 같은 seed를 전달해 H/V flip이 400/400회 함께 적용되거나 함께 생략되는 것도 관찰했다. 이는 RNG 다양성·재현성 결함의 증거이지 mIoU 하락 또는 개선의 증거가 아니다.

Model390은 기존 crop sampler·augmentation 종류·모델·loss를 유지하고, 난수 스트림을 샘플별로 결정적으로 분리한다. 각 샘플의 crop 및 augmentation seed는 `(run_seed, epoch, virtual_sample_index, stream_id)`에서 파생한 64-bit integer다. Albumentations Compose root와 각 nested transform node에도 고유 seed를 배정한다. 처치는 단순히 한 번의 난수 realization을 바꾸는 것이 아니라, legacy의 worker/epoch 간 stream 의존성도 교정한다. 관측 Δ에는 이 구조 변경과 단일 학습의 stochastic realization 차이가 함께 들어간다. 가설은 이 교정이 crop/augmentation coverage와 반복 실험 재현성을 개선할 수 있다는 것이다. 방향·크기는 사전 예측하지 않는다.

## 시작 조건과 고정 코드

- P100에서 진행 중인 Model382의 여덟 seed/fold 학습을 수정·중단하지 않는다. Model382의 사전등록 matched-count/seed 비교, Model387/388/389 locked audit, 그리고 gate를 통과한 locked branch가 있으면 그 실행까지 끝난 뒤 GPU 슬롯을 배정한다.
- Model390 코드는 P100 Model382 trainer의 정확한 source tree를 기반으로 한 별도 `model336_hrnet_rng_candidate/` 복사본이다. 활성 `/workspace/komap_model1/model336_hrnet_best`는 수정하지 않는다.
- source hash 및 이 문서의 pre-score lock receipt는 [Model390 lock](results/model390_preregistration_lock.json)이다. lock 이후 코드를 수정하면 새 버전 및 사전등록으로 분기한다.
- dependency는 smoke에서 확인한 P100 PyTorch `2.6.0+cu118`, Albumentations `2.0.8`로 고정한다. per-item 모드에서 `set_random_seed`가 없거나 transform tree의 어떤 노드도 seed할 수 없으면 즉시 실패한다(조용한 fallback 금지).
- 재개 시 `train_state.pt`의 저장 args와 현재 args를 `output_dir`·`resume` 이외 전부 비교한다. `rng_mode`, seed 또는 다른 학습 인자가 다르면 `args.json`을 덮어쓰기 전에 실패한다. Model382 runner처럼 `--resume`은 초기 실행/복구 실행 모두 전달하고, Model390은 `--rng-mode per_item`도 두 명령 모두에 명시한다.

## 실험 설계

- **대조:** Model382의 동일 seed·동일 outer fold train-OOF 예측. 그 run은 legacy RNG mode다.
- **처치:** Model390 `--rng-mode per_item`, 같은 seed·fold·train image·group assignment 및 Model382 recipe. 의도된 변경은 crop/augmentation RNG stream의 worker/epoch/sample 독립화와 그로 인한 realization이다.
- 두 arm에서 output path와 RNG mode를 제외한 모든 CLI 인자를 Model382 P100 runner와 동일하게 고정한다: P100 `cuda:0`, `UnetPlusPlus`, `tu-hrnet_w48`, ImageNet encoder weights, patch 512/input scale 2, 12 samples/image, batch 4 × accumulation 2, 4 workers, 70 epochs, AdamW LR `3e-4`, encoder LR multiplier `1.0`, weight decay `1e-4`, boundary head on/weight `1.0`, AMP 기본 on, no init-checkpoint, no CE class weights, `--val-every 0`, `--resume`. Seed·fold별 output path만 분리한다. `--resume`는 빈 output directory에서는 새 run을 시작하며, 중단 복구 시에는 저장 args 전체 일치 검사를 통과해야 한다.
- 두 arm은 같은 main seed로 model initialization, minority-coordinate subsampling, DataLoader sampler 순서를 맞춘다. per-item derivation은 worker에서만 실행하며 main Torch RNG를 소비하지 않는다. `num_workers=4`, batch/accumulation, drop-last, dataset length 및 epoch 수는 반드시 동일하게 유지한다.
- 마지막 epoch checkpoint를 쓰고 validation/test를 학습·checkpoint 선택에 쓰지 않는다. 특히 `--val-every 0`을 고정해 공식 valid quick-score 노출을 차단한다.

Model382 seed/fold별 동일 train-data root를 사용하고 output만 분리한다. 처치 실행의 전체 인자 템플릿은 다음과 같다(대조 arm은 `--rng-mode per_item`만 제거).

```bash
python -u model336_hrnet_rng_candidate/train.py \
  --data-root "$ROOT/data_model382_seed${seed}_fold${fold}" \
  --output-dir "$OUT/model390_per_item_rng/seed${seed}/fold${fold}" \
  --device cuda:0 --architecture UnetPlusPlus \
  --encoder-name tu-hrnet_w48 --encoder-weights imagenet \
  --patch-size 512 --input-scale 2.0 --samples-per-image 12 \
  --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs 70 \
  --lr 3e-4 --encoder-lr-mult 1.0 --weight-decay 1e-4 \
  --boundary-head-weight 1.0 --seed "$seed" --val-every 0 \
  --rng-mode per_item --resume
```
- 각 seed에서 네 outer fold를 모두 학습한다. seed1이 primary다. Model382 seed1 OOF와 paired 비교 후 사전 gate를 통과할 때만 seed2 replication을 시작한다. seed별 결과를 별도 보고한다.
- OOF 추론은 lock된 Model382 runner(`run_model382_seed_count_control_p100.sh`, SHA-256 `753c1ae26cdd663aa104fa6143a725c325fb48fe7a2b7209a878c3120c101f69`)와 같은 `evaluate_image_mean.py` 구현/hash 및 고정 옵션을 쓴다: `--split valid` 경로는 official valid가 아니라 해당 fold의 held-out train image/mask symlink이며, patch 512, overlap 170, input scale 2, flip4 TTA, antialias off, batch 2, `cuda:0`. Model382 runner는 이 옵션 중 기본값인 patch/overlap/scale/tta를 생략하므로 Model390 실행 명령에는 이를 명시해 고정한다. `valid_summary.json`의 이름을 근거로 official validation이라고 해석하지 않는다.
- 구체적으로 candidate의 같은 `evaluate_image_mean.py`에 `--checkpoint "$run/last.pt" --data-root "$data_root" --split valid --output-dir "$oof" --device cuda:0 --batch-size 2 --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4`를 준다. `--no-tta`와 `--downsample-antialias`는 주지 않는다(false). `data_root/valid`는 locked `cv_data/oof_fold${fold}` symlink로 확인한다.
- OOF 산출은 held-out train image에 한 번씩만 수행하며 다른 fold의 예측으로 대체하지 않는다. 이미지/group/fold identity, native shape, checkpoint·prediction hash가 Model382 manifest와 맞지 않으면 중단한다. Evaluator source hash와 이 inference 설정을 결과 전 lock receipt에 보존한다.
- candidate OOF 디렉터리마다 `model390_manifest.json`을 기록한다. 필수 필드는 model, seed, fold, `epochs=70`, `rng_mode`, metric, checkpoint 및 probability archive SHA-256, Model390 lock JSON SHA-256, 고정 inference options, 저장된 `args.json` 전체다. 모든 candidate 산출물의 mtime은 Model390 lock 시각 이후여야 한다. 사후 evaluator는 candidate/evaluator/smoke/protocol 소스 hash와 Model382 비교 코드·사전등록 receipt·reference runner hash를 검증한다. 또한 Model382 receipt에 고정된 네 분석 파일의 hash를 재검증하고, manifest/hash/args 및 Model382 manifest를 확인한 뒤 70 image/59 group coverage를 재계산한다.

## 지표·불확실성·판정

- 공식 지표는 image-wise mean mIoU: 각 이미지에서 4개 class IoU를 평균한 뒤 70 train images/59 specimen groups 평균을 내며, empty union은 0으로 처리한다. class별 safety delta도 empty union을 0으로 두고 각 class IoU를 70 images에 대해 평균한다. Step4는 image stem의 `_step(4)_` 패턴으로 사전 식별하며, 해당 image들의 image-wise mIoU 평균을 쓴다. Step4 이미지가 없으면 gate를 계산하지 않고 중단한다.
- paired 차이는 같은 이미지·fold·seed의 처치−대조로 계산한다. group bootstrap은 fold 층화 specimen-group resampling, 20,000회, bootstrap seed `390`으로 고정한다. seed가 두 개일 때 같은 group draw에 그 group의 두 seed 예측을 함께 넣어 seed 두 개를 독립 specimen처럼 세지 않는다. group CI는 고정한 seed 설정 아래의 data-sampling uncertainty이지 학습-seed 모집단 CI가 아니다.
- Model390 seed별 Δ는 Model382의 기존 same-recipe seed1/seed2 대조 차이와 나란히 기술 보고해 학습-seed noise 맥락을 보인다. 이 참고 비교는 gate나 사후 임계값을 바꾸지 않는다. group bootstrap CI는 학습 stochasticity, cuDNN/AMP 비결정성, 기존 실험 선택 편향을 포함하지 않는다.
- **seed1 continuation gate:** Δ `≥+0.50%p`, paired group-bootstrap 95% CI 하한 `>0`, 네 fold 중 최소 3개 개선, 최악 fold `≥−0.30%p`, 모든 class delta `≥−0.50%p`, Step4 delta `≥−0.50%p`. 하나라도 실패하면 seed2 Model390 학습 및 valid 평가를 하지 않는다.
- **최종 adoption gate (seed2까지 실행된 경우):** 두 seed 각각 point delta `>0`; 두 seed equal-weight 평균 Δ `≥+0.50%p`; paired group-bootstrap CI 하한 `>0`; seed별 네 fold 중 최소 3개 개선; 최악 fold·class·Step4 guardrail은 위 문턱을 각각 만족해야 한다. 실패하면 점수 향상 recipe로 채택하지 않는다.
- gate 통과는 독립 성능 증명이 아니다. 통과한 고정 recipe에 한해 full-train 학습 및 official valid 20장 단회 평가를 계획한다. valid 결과로 seed·threshold·tile·ensemble을 다시 선택하지 않는다. test는 최종 제출 결정 때까지 열지 않는다.
- 단계 evaluator는 [model390_per_item_rng_oof.py](model390_per_item_rng_oof.py)다. `--stage seed1`은 continuation gate 결과를 출력한다. `--stage final`은 seed1 결과 JSON을 `--seed1-result`로 요구하고, 해당 gate 및 같은 lock hash를 검증하지 않으면 seed2 비교를 수행하지 않는다. `valid_evaluation_authorized`가 true일 때만 단회 valid 평가를 고려한다.

## Smoke 범위와 미확인 사항

P100 환경에서 synthetic-only smoke가 (a) candidate legacy와 원 P100-base dataset의 bitwise equality, (b) per-item output의 worker 수 0/2/4·batch 크기·처리 순서 불변성, (c) epoch 변경과 동일 seed/epoch data replay, (d) 독립 H/V flip gate 네 조합 및 full transform-tree 고유 seed, (e) synthetic CPU train.py epoch, completed-epoch resume state 복원 및 RNG mode mismatch 거부를 통과했다. 임시 합성 데이터와 임시 checkpoint만 사용했다. 이 trainer smoke는 미완료 epoch 중단/재개 또는 전체 optimizer 궤적의 bitwise 재현 시험은 아니다. mIoU, 실제 crop coverage, GPU 학습 안정성 및 성능 향상은 측정하지 않았다.

재현: [candidate source](../model336_hrnet_rng_candidate), [data-pipeline smoke](rng_per_item_candidate_smoke.py), [paired OOF evaluator](model390_per_item_rng_oof.py), [OOF evaluator synthetic smoke](model390_per_item_rng_oof_smoke.py), [초기 worker-state smoke](worker_rng_reseed_smoke.py).
