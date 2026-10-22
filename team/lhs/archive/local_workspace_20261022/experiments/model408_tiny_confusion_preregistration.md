# Model408: tiny-GT confusion hinge

## Motivation

Model407 reduced 1–63 native-pixel predicted-component FP mass, but paired OOF
mIoU fell by 0.0868 pp in all four folds. The candidate also reduced Al3Ni and
Eutectic-Si tiny GT-component hit rates and shifted 1–3px GT pixels toward Al.
All 70 training images contain both target classes, and a fixed sample of
1,024 post-augmentation training crops per target class contained no absent
class crops. Therefore neither image-absence nor crop-absence FP gating is the
next treatment.

The earlier locked confusion audit shows a specific direction: Al3Ni tiny GT
pixels are mainly confused with Al, while Eutectic-Si tiny GT pixels are
confused with Al3Ni and Al. Model403's global supervised contrastive objective
sampled decoder-grid labels and did not isolate native-tiny GT components.

## Fixed treatment

- Start from each fixed Model392 seed-1 fold checkpoint; use the existing
  Model404 continuation recipe (HRNet-W48 U-Net++, patch 512, width-normalized
  input scale 2, same per-item crop/augmentation stream, optimizer, schedule,
  and eight epochs).
- On the post-augmentation CPU mask, reuse the locked Al3Ni/Eutectic-Si GT
  component bins `[4,16)`, `[16,64)`, `[64,256)` pixels, which approximately
  cover native component sizes 1–64px under the nominal 2× scale. Crop-border
  components remain excluded as in the prior loss.
- For each selected GT component pixel, apply a zero-margin hinge only when a
  fixed confuser logit is at least the true-class logit:
  `relu(max(confuser_logits) - true_logit)`. Al3Ni's confuser is Al; Eutectic
  Si's confusers are Al and Al3Ni. Average pixel hinges within each component,
  then average components within crop × target-class × size-bin cells.
- Correctly classified pixels and pixels outside selected tiny GT components
  receive no auxiliary gradient. No prediction-side false-positive term,
  confidence threshold, margin search, sampler change, new head, or inference
  routing is added. Existing CE+Dice and boundary losses remain unchanged.
- Calibrate a single fold-specific coefficient to 10% median decoder-gradient
  norm using 32 outer-train crops/fold, zero optimizer steps. No held-out groups,
  official validation, or test data are read.

## Evaluation and adoption

Use fixed 4-fold/70-image/59-group train OOF and the locked Model406 v3
compute-matched continuation control. Score the official image-wise mean of
four class IoUs with empty-union=0 and 20,000 paired specimen-group bootstrap
draws. The pre-registered gate is: at least +0.20pp, paired CI lower bound >0,
at least 3/4 positive folds, worst fold >=−0.30pp, Step4 >=−0.30pp, no target
class IoU regression, all-class and Model392 parent safeguards, no decrease in
tiny/small GT component hit rates, and no increase in unmatched 1–63px
predicted components for either target class. A pass only permits an
independently locked seed-2 grouped-OOF confirmation. A failure is NO-GO; no
official validation/test or post-hoc margin/weight search is authorized.
