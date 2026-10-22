# Model392 사전등록: HRNet per-image pixel-scale 정규화

## 가설과 제한

Model391 train-mask-only 분석은 native width와 상 경계 run 길이의 step-adjusted 기울기가 Al3Ni 1.217 (59-group bootstrap 95% CI `[0.762,1.634]`), Eutectic Si 0.838 (`[0.605,1.072]`)로 사전 기준을 통과했다. 그러나 해상도 네 strata의 specimen-group memberships가 `16+15+21+7=59`여서 각 group은 단 하나의 width에만 속했다. 그러므로 그 결과는 morphology와 분리되지 않은 between-group association이지, 배율이 물리적 pixel scale을 일으킨다는 증거가 아니다. [Model391 interpretation addendum](model391_interpretation_addendum.md)을 포함해 이 한계를 전제로 한다. Model392는 이를 바로 성능/일반화 OOF에서 직접 반증할 저자유도 실험이며, 효과를 미리 주장하지 않는다.

## 고정 개입

- 대조군은 Model382의 같은 seed·outer fold train-only OOF다. 처치군은 Model392 `--normalize-width` HRNet-W48 + U-Net++다. 한 가지 의도된 변경은 이미지별 pixel scale이다. `--rng-mode legacy`로 난수 처리 방식은 Model382와 같게 둔다.
- 각 outer fold에서 `target_width`는 오직 그 fold의 `train/images` native widths의 mode로 정한다(동률이면 더 작은 width). Held-out OOF image width나 mask는 target 선택에 사용하지 않는다. Full-train 재학습 시 같은 규칙을 전체 train 이미지에 적용한다. 현재 full-train mode는 1278px이지만 fold target은 training subset에 따라 다를 수 있으므로 각 checkpoint/manifest에 반드시 기록한다.
- 학습 image 폭을 `W`, target을 `T`, 기존 base `input_scale=2.0`으로 둘 때 effective scale은 `2*T/W`; crop native side는 `round(512/(2*T/W))`다. Image/mask는 같은 crop 좌표를 쓰며 image는 cubic, mask는 nearest로 512 patch에 맞춘다. Rare-coordinate sampler/augmentation/loss/model은 유지한다.
- 추론도 checkpoint에 저장된 동일 T와 원본 image width로 effective scale을 계산한다. native tile은 `round(512/effective_scale)`이고 base overlap 170은 tile/base-tile 비율(170/256)이 유지되도록 비례 조정한다. Probability map은 원본 resolution에 반환한다. 저장된 preprocessing metadata와 명시 CLI option이 다르면 즉시 중단한다. Fold ensemble은 각 checkpoint의 자기 target width로 별도 추론한다.
- `--val-every 0`은 필수다. `--normalize-width`와 다른 값은 trainer가 거부한다. 매 fold 마지막 epoch checkpoint만 사용한다. Model382와 동일한 나머지 recipe: P100 `cuda:0`, `UnetPlusPlus`, `tu-hrnet_w48`, ImageNet, patch 512, 12 samples/image, batch 4 × accumulation 2, 4 workers, 70 epochs, AdamW LR `3e-4`, encoder multiplier 1.0, weight decay `1e-4`, boundary head weight 1.0, AMP on, no CE class weights/init checkpoint, `--resume`, seeds 1/2.
- 새 GPU 작업은 기존 Model382 matched-count/seed 비교, Model387/388/389 locked audits와 해당 gate가 끝난 뒤에만 고려한다. 이 사전등록은 지금 학습을 시작하라는 승인이 아니다.

## 평가와 순차 gate

- 고정 4-fold specimen-group split의 70 train images/59 groups에 exactly-once OOF를 만든다. `data_root/valid`는 fold의 held-out train group symlink다. Official valid/test는 보지 않는다. Paired metric은 각 이미지의 4개 class IoU를 평균하고 images를 평균한 공식 image-wise mIoU, empty union=0이다. Group bootstrap은 fold-stratified 59-group resampling, 20,000회, seed 390이다. Two-seed final CI는 이미지마다 seed1/seed2의 paired delta를 먼저 평균하고, 동일한 fold-stratified group draw를 적용한다.
- Primary는 seed1의 Model392 대 같은 fold Model382 seed1이다. seed1 continuation gate는 Δ `≥+0.50%p`, group bootstrap CI lower `>0`, fold 최소 3/4 개선, worst fold `≥−0.30%p`, 모든 class 및 Step4 `≥−0.50%p`다. 하나라도 실패하면 seed2 학습 및 valid 측정을 하지 않는다.
- seed1 통과 때만 seed2를 4 folds 학습한다. 최종 gate는 두 seed point delta 모두 `>0`, equal-weight mean delta `≥+0.50%p`, paired group bootstrap CI lower `>0`, 그리고 seed별·pooled fold/class/Step4 safety 조건 통과다. 실패 시 성능 recipe로 채택하지 않는다.
- Final gate를 모두 통과해야 official valid 20장 단회 평가를 고려한다. Valid로 target width, seed, threshold, tile, TTA 또는 ensemble을 고르지 않는다. Test는 최종 제출 때까지 열지 않는다. 모델/요약 출력은 덮어쓰지 않는다.

## 산출물 무결성

각 OOF manifest에는 model/seed/fold/epoch, `rng_mode=legacy`, metric, fold-specific `target_width`, full training args, preprocessing/inference settings, checkpoint 및 OOF archive SHA-256, Model392 lock SHA-256을 기록한다. Evaluator는 Model392 source/protocol/synthetic smoke, Model391 result 및 lock, Model382 helper/receipt/reference runner hashes, Model382/392 manifests, args, artifact mtimes, exact IDs/shapes/probability normalization을 확인한다.

후보 source: [isolated tree](../model336_hrnet_widthnorm_candidate). Synthetic CPU/P100 smoke: [candidate smoke](model392_widthnorm_candidate_smoke.py), [OOF evaluator smoke](model392_widthnorm_oof_smoke.py). Per-fold artifact validation/manifest writer: [recorder](model392_record_fold.py). Paired analysis: [OOF evaluator](model392_widthnorm_oof.py). Pre-score hashes: [lock receipt](results/model392_widthnorm_preregistration_lock.json).
