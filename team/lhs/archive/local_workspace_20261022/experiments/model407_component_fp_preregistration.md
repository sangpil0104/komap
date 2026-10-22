# Model407: GT-component recall with prediction-component FP suppression

## Motivation

Model406 v3 increased GT component recall for Al3Ni and Eutectic Si in all
native-size bins, but its top-0.1% pixel hard-negative term did not suppress
unmatched tiny predicted components. Image-wise OOF gain was only +0.01985 pp
and the preregistered island guard failed. Model407 keeps the same balanced GT
component-recall term and replaces only the pixel hard-negative term with a
prediction-connected-component penalty.

## Fixed treatment

- Classes: Al3Ni (1) and Eutectic Si (3), with the existing Model404 v4
  balanced GT component-recall loss unchanged.
- On each training batch, obtain a detached four-class argmax partition.
  Label 8-connected components for each target class on CPU.
- A predicted component is eligible for the negative term only if it does not
  touch the same-class one-pixel GT halo and does not touch the crop border.
  The halo protects annotation/phase boundaries; border exclusion protects
  components truncated by crop sampling. No confidence threshold or predicted
  area cutoff is used.
- For each eligible component, penalize
  `-log(1 - mean(p_target over component))`. Sum component penalties within
  each crop-by-target-class cell and average over all batch-by-class cells,
  including empty cells. This is deliberately count-sensitive: crops with
  more unmatched islands contribute more negative-loss pressure.
- The joint auxiliary is balanced GT-component recall plus this prediction-CC
  penalty. It replaces Model406's top-0.1% pixel hard-negative term; no other
  loss, architecture, sampling, augmentation, optimizer, schedule, or TTA
  change is allowed.

## Calibration and comparison

- Calibrate one fold-specific coefficient to a 10% median decoder-gradient
  norm ratio using 32 batch-one crops from each outer-train split only.
  Optimizer steps are zero; held-out groups, official validation, and test are
  never read during calibration.
- Start each fold from its fixed Model392 seed-1 checkpoint. Use the same four
  specimen-group folds, per-item seeded crop/augmentation stream, patch 512,
  width-normalized input scale 2, batch 4 with accumulation 2, AdamW learning
  rates, and eight epochs as Model406 v3.
- Reuse the hash-locked Model406 v3 matched-control OOF maps as the primary
  paired baseline. Also report the new candidate versus the completed Model406
  pixel-hard-negative candidate; this is descriptive and cannot rescue a
  failed primary gate.
- Evaluate 70 training images / 59 specimen groups with official image-wise
  mean of four class IoUs and empty-union=0, plus 20,000 paired specimen-group
  bootstrap draws. Official validation and test remain sealed.

## Adoption gate

Require every Model406 v3 gate against the fixed Model406 v3 control: at least
+0.20 percentage points image-wise mIoU; paired group-bootstrap 95% CI lower
bound > 0; at least 3/4 positive folds; worst fold >= -0.30 pp; Step4 >= -0.30
pp; no target-class IoU regression; all-class and parent safeguards; and no
increase in unmatched 1–63 native-pixel predicted components for either target
class. Report class IoUs, Step4, tiny/small/medium GT-component recall,
unmatched predicted components and target FP pixels. A pass authorizes only a
separately locked independent-seed grouped-OOF confirmation.

## Known risks and stop rule

The predicted partition is nondifferentiable and requires an argmax transfer
to CPU; gradients flow only through probabilities within selected components.
All predicted component sizes are included because crop resize, width
normalization, affine transforms, and elastic deformation make a fixed
augmented-grid size cutoff a poor proxy for native area. The count-sensitive
penalty can over-suppress rare target phases; class-IoU, component-recall, and
image-mIoU gates are therefore mandatory. Do not search a coefficient, radius,
connectivity, threshold, or blend. If any mandatory gate fails, mark NO-GO and
do not look at official validation/test to select a rescue.
