# Model382 secondary sensitivity: equal mean of the two fixed seeds

## Status and relationship to Model382 primary

This is a **secondary, development-only** analysis, added after the Model382 seed1 OOF files had been generated but before any seed1/seed2 OOF score values were inspected. The outputs and hashes existed at lock time; only file presence, sizes, hashes, training logs, and protocol/manifests were inspected. It does not replace, amend, or rescue the locked Model382 primary comparison.

The primary comparison remains unchanged: A/S1/S2/B/C (and separately prelocked E5), using same-fold Model374/376/377 and Model382 OOF maps. If historical baseline maps remain unavailable, that primary result is reported as **not evaluable because required inputs are missing**. No validation/test prediction or label may substitute for them.

This secondary asks only whether a fixed 1:1 soft-probability mean of Model382 seed1 and seed2 OOF predictions beats each singleton. It is useful for quantifying same-recipe averaging when historical comparison arms cannot be reconstructed. It is conditional development evidence—not an independent test, not a Model379 re-evaluation, and not a change to its adoption result.

## Fixed data and prediction rules

- Exactly the existing 70 train images / 59 specimen groups and the four fixed grouped OOF folds. Each image is evaluated only with probabilities from the held-out-fold checkpoint.
- Use the two Model382 HRNet-W48/U-Net++ checkpoints already specified in its locked protocol. Verify their fold, seed, 70 epochs, checkpoint hash, OOF archive hash, image IDs, tensor shape, finite values, probability range, and per-pixel normalization through the original Model382 verifier.
- For each pixel, `E = 0.5 * softmax(seed1) + 0.5 * softmax(seed2)`, followed by argmax. No learned weights, temperature, threshold, TTA, post-processing, or validation/test access.
- Metric is the official image-wise mean of four class IoUs, with empty union scored as zero. Compare E separately against seed1 and seed2; never select the better singleton as a sole reference and never compare against the mean of singleton scores.
- Group-bootstrap paired image deltas, stratified within each outer fold, 20,000 draws, fixed RNG seed 3822. Resample specimen groups and retain all images in each sampled group. Also report fold, class, Step4, and A71-excluded sensitivities.
- Require exactly 70 unique OOF image IDs, 59 groups, no image duplicates, and no group crossing folds. No valid/test path is accepted by the evaluator.

## Frozen gates

The secondary gate passes only if **all** conditions hold:

1. E improves the image-wise mean mIoU point estimate by at least `+0.20 percentage points` versus **each** of seed1 and seed2.
2. The paired fold-stratified specimen-group bootstrap 95% CI lower bound is strictly above zero for **both** comparisons.
3. For each comparison, no class mean IoU drops by more than `1.0 percentage point`.
4. In at least 3 of 4 folds, E beats both singleton seeds.

Passing this gate is only a promising development signal. A follow-up/full-data candidate is not automatic. For adoption consideration, both E-vs-seed point estimates must additionally be at least `+0.50 percentage points`, and the existing independent training/validation policy still applies. Failure means do not expand seed count or search ensemble weights from this result.

## Reproducibility and leakage controls

- This secondary specification and code are hash-locked before seed2 OOF outputs exist. At evaluation, every seed2 checkpoint, OOF manifest, and OOF archive must have modification times later than the secondary lock. Seed1 files may predate this secondary lock, but must pass the original Model382 lock and verifier; their OOF score values must remain unopened until this lock is recorded.
- All masks/probabilities come from fixed train-OOF only. `valid.zip`, `test.zip`, and their labels/predictions are not inputs.
- It does not access official validation/test and does not modify the original Model382 primary/e5 protocol or lock.

## Reproduction

```bash
python experiments/model382_two_seed_mean_sensitivity.py \
  --cv-root /path/to/data_model322_group_cv \
  --model382-root /path/to/outputs/model382_seed_count_control \
  --output experiments/results/model382_two_seed_mean_sensitivity.json
```

The analysis runs on CPU. A synthetic-only smoke test is `experiments/model382_two_seed_mean_sensitivity_smoke.py`.
