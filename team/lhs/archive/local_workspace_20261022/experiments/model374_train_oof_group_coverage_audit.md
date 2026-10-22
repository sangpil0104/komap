# Model374 train-group coverage association audit (exploratory)

## Purpose and status

This CPU-only descriptive audit asks whether held-out training specimen groups with lower Model374 OOF image mIoU are also farther from the training specimens in fixed input/mask feature spaces. It is a zero-GPU triage suggested by Claude Code Opus before considering a costly learning-curve run.

**This is post-hoc and selection-conditional.** Model374 OOF image/group scores and related error summaries were already inspected before this specific association was formulated. No p-value, confidence interval, or association from this audit is confirmatory, causal, or an automatic gate to start/stop training. It does not replace the locked Model382, 387, 388, or 389 analyses.

## Inputs and exclusions

- `train.zip`: the 70 original training images and their masks.
- `model374_oof_boundary.json`: existing exactly-once, four-fold train-OOF per-image mIoU, group, and fold metadata.
- No official valid/test images, masks, predictions, or scores; no Model382 OOF values.
- Each specimen group is held out as a unit. Its nearest reference set is the other-fold training groups; group overlap across folds is checked and rejected.

## Fixed feature and outcome definitions

- Unit of analysis is the specimen group. Its score is the unweighted mean of its image-wise four-class mIoU values (empty union=0).
- **Appearance distance:** nearest other-fold training-group distance between group-mean vectors of the existing 15 grayscale intensity/texture features defined in `valid_test_similarity_audit.py`. Robust feature scaling is fitted only on the training groups within that outer fold; intensity and texture feature blocks keep the existing fixed block weights.
- **Step/resolution distance:** nearest-group distance, defined as the mean of the total-variation distances between group-level step histograms and resolution histograms.
- **Class-mix distance:** nearest-group total-variation distance between group-mean four-class GT pixel fractions. This is a descriptive case-mix measure, not an input-domain feature; held-out masks are used only because this is an OOF diagnostic.
- For each feature family, calculate Spearman correlation between nearest-training-group distance and held-out group mIoU separately within each outer fold, then report the unweighted mean of the four fold correlations. Negative means farther groups scored lower.
- A 20,000-draw bootstrap resamples specimen groups within each fixed held-out fold and recomputes the mean fold correlation. The interval is conditional on the existing fitted OOF models and omits training/subset/fold-model variance; outer training sets overlap.

## Interpretation limits

The analysis can describe whether measured input, metadata, or label-composition dissimilarity co-occurs with lower OOF group score. It cannot establish that more training groups would improve a refit, separate specimen diversity from pixels/patches, identify label noise, estimate valid/test performance, or show that the nearest-neighbor feature space is sufficient. A weak/noisy association is not evidence that dataset size is irrelevant. Any group-subset training experiment needs its own pre-result protocol and compute/decision justification; this audit alone never triggers one.

## Reproduction

```bash
python3 experiments/model374_train_oof_group_coverage_audit.py \
  --train-zip train.zip \
  --oof-json experiments/results/model374_oof_boundary.json \
  --bootstrap-draws 20000 \
  --output experiments/results/model374_train_oof_group_coverage_audit.json
```

Synthetic mechanics check:

```bash
python3 experiments/model374_train_oof_group_coverage_audit_smoke.py
```

## Post-hoc followups

After the primary exploratory association was calculated, the following fixed descriptive checks were added. They are not preregistered or confirmatory and do not decide any GPU run:

```bash
python3 experiments/model374_train_oof_group_coverage_followup.py \
  --coverage-json experiments/results/model374_train_oof_group_coverage_audit.json \
  --oof-json experiments/results/model374_oof_boundary.json \
  --output experiments/results/model374_train_oof_group_coverage_followup.json

python3 experiments/model374_target_group_coverage_audit.py \
  --train-zip train.zip --valid-zip valid.zip --test-zip test.zip \
  --coverage-json experiments/results/model374_train_oof_group_coverage_audit.json \
  --output experiments/results/model374_target_group_coverage_audit.json
```

The target-side script reads **only image members** from the three ZIPs; it does not open valid/test masks, prediction files, or target scores. Each valid/test specimen group's fixed grayscale-feature distance to each Model374 outer-fold training pool is percentile-ranked against that fold's held-out OOF group-distance distribution. It reports a same-prefix-in-train sensitivity; percentile is an input-domain proxy, never an mIoU estimate. The followup reports leave-two-farthest-group correlations by fold, per-class association, and a hypothetical quartile scenario; none is causal or an achievable model gain.
