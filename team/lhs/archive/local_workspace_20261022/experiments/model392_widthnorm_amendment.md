# Model392 protocol amendment (before candidate training or score access)

This amendment supersedes the staged seed1/seed2 decision rule in the original
[Model392 preregistration](model392_widthnorm_preregistration.md). The original
receipt remains unchanged for provenance; no Model392 training, OOF score, or
official validation/test result existed when this amendment was prepared.

## Why amend

Claude Code Opus's independent review found no confirmed metric or tiling bug,
but identified that gating seed2 on seed1's score and then using the equal-weight
two-seed average as the final estimate conditions that estimate on seed1's
success. That creates an upward-selected point estimate and an unadjusted CI.
The final two-seed mean also reuses the same 70 groups. Therefore both seeds
will be completed before any candidate OOF score is opened or the final gate is
computed. There is no interim seed1 futility stop.

The final estimand is the fixed, equal-weight two-seed Model392 recipe on the
70-image/59-specimen grouped OOF set. For each image, average the two seeds'
four-class image mIoU; then average images. The paired fold-stratified
specimen-group bootstrap (20,000 draws, seed 390) resamples the groups once
after averaging the seed-specific image deltas. Its interval is conditional on
these two trained seeds, fixed folds and checkpoints; it does not estimate
training-seed variance or establish performance on unseen specimens by itself.
Report both seed deltas separately, the equal-weight mean, and their difference.

## Confirmatory gate and permitted validation

After all eight runs (seeds 1/2 × folds 1–4) and exactly-once OOF inference are
complete, run the analysis once. Continue to full-train fitting only if all
conditions hold:

- each seed's paired image-mean delta versus the same-seed/fold Model382
  reference is positive;
- the equal-weight two-seed delta is at least `+0.50%p` and its paired
  specimen-group bootstrap 95% CI lower bound is above zero;
- for each seed and the pooled result, at least 3/4 folds improve, no fold is
  below `−0.30%p`, every class is at least `−0.50%p`, and Step4 is at least
  `−0.50%p`.

Fold/class/Step4 and native-width stratum results are descriptive safety
breakdowns, not additional tuning opportunities. Native-width strata,
fold-specific target-width counts, and the in-fold target-width images are
reported as secondary diagnostics only. In particular, 1278px is an internal
preprocessing control only in folds whose training-only mode is 1278px.

Only after this OOF gate passes, train full-train checkpoints for both seeds,
average their probability maps 1:1, and make one official-valid evaluation
using the empty-union=0 image-wise four-class mIoU. No threshold, scale, TTA,
blend weight, or postprocessing is tuned on valid. Test remains unopened.
The full-train runner is [P100 sequential runner](../run_model392_full_train_p100.sh);
it requires the passing two-seed OOF receipt, refuses to overlap Model382, uses
only `data/train` with `val_every=0`, and records seed-specific resume receipts.
The separate [one-shot evaluator](model392_official_valid_once.py) has no test
split option and refuses missing/mismatched checkpoints, a failed gate, or an
existing output directory. Its synthetic empty=0, gate, ID, and run-receipt
smoke is [here](model392_official_valid_once_smoke.py).

After all eight fold manifests/checkpoints/OOF archives exist, write the single
confirmatory receipt to the exact path expected by both downstream guards:

```bash
python experiments/model392_widthnorm_oof.py \
  --cv-root /workspace/komap_model1/data_model322_group_cv \
  --model382-root /workspace/komap_model1/outputs/model382_seed_count_control \
  --model392-root /workspace/komap_model1/outputs/model392_widthnorm \
  --output /workspace/komap_model1/experiments/results/model392_widthnorm_two_seed_confirmatory.json

bash /workspace/komap_model1/run_model392_full_train_p100.sh

python /workspace/komap_model1/experiments/model392_official_valid_once.py \
  --project-root /workspace/komap_model1
```

## Geometry and execution integrity

Before candidate training, run the locked [fold geometry audit](model392_widthnorm_fold_geometry_audit.py).
It may read only `train_fold{k}/images` and `oof_fold{k}/images` filenames and
image dimensions from the fixed grouped split; it never reads masks, labels,
predictions, OOF probabilities, official valid, or test. Each target width is
the mode of that fold's training image widths (ties to smaller width). Report
training and held-out width counts plus resulting scale, native crop/tile size,
and overlap. Do not change the rule after seeing the table.

Resolve project/data/output paths to absolute paths before recording args or
launching training. Every fold must have a lock-bound run receipt created before
its first training attempt; it records start time, exact args, lock hash, and
resume epochs/checkpoint hashes. Resume only from that receipt and preserve the
attempt history. Recorder and evaluator reject naive timestamps, path/config
mismatches, incomplete epochs, stale artifacts, wrong checkpoint preprocessing,
or inference settings other than batch 2 / `cuda:0` / flip4 / antialias off.

The candidate is a geometric preprocessing intervention, not calibrated
physical magnification. It changes native context, crop fraction, rare-pixel
jitter and inference tiling, and can alter aliasing. Any observed effect is
attributed only to this complete width-normalized recipe. The Model391
between-specimen association is not causal evidence.

The image-metadata-only fold audit found training-mode targets `1230px` for
fold 1 and `1278px` for folds 2–4. Held-out images whose width equals their
fold's train-mode target number `2, 7, 4, 6` respectively (19/70 total). The
675px strata use native crops/tiles of `140px` in fold 1 and `135px` in the
other folds; 1440px strata use `300px` and `288px`. This confirms that the
registered intervention is a substantial context/scale change, not a tiny
calibration. These metadata are descriptive; targets and recipe stay fixed.
See the [locked-scope geometry output](results/model392_widthnorm_fold_geometry_audit.json).

No training starts until the currently locked Model382 matched-count/seed
analysis and Model387/388/389 audit gates are complete. The amended scoring
lock is [recorded separately](results/model392_widthnorm_amendment_lock.json);
the original pre-score lock is preserved unchanged.
