# Model392 auxiliary boundary-head localization probe

## Objective and scope

Measure whether the already-trained, class-agnostic boundary auxiliary head in
Model392 localizes annotated phase boundaries on specimen-group-held-out
training images. This is a diagnostic for whether a later class-specific
boundary-to-interior direction head is worth having Claude Code review. It is
not a segmentation score and cannot by itself authorize a new training run.

Use all 70 train images and 59 specimen groups, with each image predicted by
both Model392 seeds using the checkpoint for its fixed held-out fold. No model
is trained. Only the grouped-OOF training images/masks and the eight
corresponding Model392 checkpoints are read. Official validation and test
data, labels, probabilities, scores, or paths are excluded.

## Frozen inference and target

- Rebuild the original Model392 HRNet-W48/U-Net++ with its saved
  boundary_head; require the original P100 Model392 source/runtime and
  sequence-closure locks, 70-epoch run receipts, fold manifests, checkpoint
  hashes, and train/OOF group disjointness to verify.
- Use each fold checkpoint's saved width target, patch size 512, base input
  scale 2.0, ImageNet RGB normalization, and flip4 TTA. Predict sigmoid
  boundary logits; inverse-transform each flip and average the four scores.
- Tile each entire held-out image at its native resolution. Derive tile size
  from the checkpoint's effective scale. Use a fixed 25% tile overlap and the
  existing Gaussian center-weighted stitching rule. This overlap is fixed for
  diagnostic efficiency; it differs from Model392's denser semantic OOF
  inference and the result must not be described as Model392's official
  inference output.
- Ground truth is the same class-agnostic target as the trained auxiliary
  loss: gt_boundary_band(mask, num_classes=4, width=1), i.e. a pixel is
  positive when its 3x3 neighborhood contains more than one phase.
- No score threshold is selected. Per-image average precision (AP) and the
  boundary-positive prevalence are the primary threshold-free quantities.
  Average per-image results within specimen group before group-macro
  aggregation. Report both seeds separately and together. Use 20,000
  specimen-group bootstrap draws with seed 392081 for the group-macro AP and
  paired AP-minus-prevalence interval.
- Report coverage, positive/negative pixel support, fold/seed summaries, and
  per-image AP/prevalence. Do not calculate or report mIoU from this probe.

## Decision rule and limits

Record LOCALIZATION_SIGNAL only if the 95% specimen-group bootstrap lower
bound for group-macro (AP - prevalence) is above zero, both seed means are
above their prevalence baselines, and at least three of four fold means are
above baseline. Otherwise record NO_LOCALIZATION_SIGNAL. This rule only
decides whether the direction-head idea merits a separate Claude review; it
does not authorize training or valid-set use.

The probe is selection-conditional because the auxiliary head and boundary
hypothesis were motivated by earlier train-OOF analyses. AP measures
class-agnostic annotation-boundary localization, not whether a direction
prediction corrects semantic errors or raises competition mIoU. A positive
result must be followed by a preregistered, matched grouped-OOF experiment
and a separate Claude Code review before training.

## One-shot provenance

Before inference, create and verify
experiments/results/model392_boundary_head_localization_lock.json. It binds
this protocol and evaluator source, the existing Model392 locks and source,
all used checkpoint/receipt/manifest hashes, and every consumed train image
and mask hash. The evaluator uses an exclusive claim and refuses overwrite or
rerun after a claimed attempt.
