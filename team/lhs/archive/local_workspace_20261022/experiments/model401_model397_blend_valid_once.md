# Model401: one-shot valid evaluation of the frozen Model397 blend

Status: `APPROVED_FOR_ONE_SHOT_VALID_EVALUATION` by independent Codex review.
Claude was not called because its usage is exhausted. This protocol does not
claim Claude endorsement.

## Why this is a new protocol

Model397's original protocol withheld its valid stage because the standalone
Model392 gate failed. However, the fixed Model397 blend itself passed its
own train-only grouped-OOF screen: `79.985862%`, with `C-H=+0.7773 pp`
(group-bootstrap 95% CI `[+0.3861,+1.2204]`), `C-B4=+0.8626 pp`
(`[+0.5855,+1.1562]`), positive gains in all four folds and all four
classes, and a largest-group gain share of `17.49%`. This new protocol
supersedes only the old prohibition on measuring this already-frozen blend;
it does not alter Model397's OOF record or gate.

## Frozen candidate and inference

For each valid image, calculate

`P = 0.50 * B4_4fold + 0.25 * HRNet_seed1_4fold + 0.25 * HRNet_seed2_4fold`,

then apply `argmax`. Use the existing Model329 four-fold B4 probability
archive. For HRNet, use all eight Model392 fold checkpoints. Each checkpoint
uses the exact preprocessing metadata saved with it (512 patch, base input
scale 2.0, fold-local target width, overlap 170 scaled by the saved pixel
scale, flip4 TTA, Gaussian probability stitching). Average four folds within
each seed before applying the fixed family weights. Do not reuse Model400's
HRNet maps: those came from Model382, not width-normalized Model392.

The input set is exactly the 20 official-valid images. Inference must not
decode valid masks. Complete, hash, and verify all 20 maps before the scorer
reads labels. No threshold, weight, calibration, routing, TTA, or
post-processing search is allowed. Do not read test files or labels.

## Metric and interpretation

Use official image-wise four-class mIoU: compute four per-image class IoUs,
set empty-union IoU to zero, average classes within each image, then average
the 20 image scores. Report the four class means, all per-image scores, and
the difference from Model364 (`79.9429727%`).

This is one additional look at an adaptively reused 20-image validation set.
The OOF estimand uses one held-out checkpoint per image and seed, while this
valid candidate averages four fold checkpoints per seed. A result is therefore
selection-conditional development evidence, not an independent test estimate.
Whether it reaches 80% is a descriptive validation result only; no follow-up
valid tuning is authorized by this protocol.

The one-shot lock binds this protocol, evaluator and smoke test, Model397's
frozen OOF lock/result, Model329's existing valid archive and receipts, all
eight Model392 checkpoints and preprocessing metadata, exact valid IDs/images
and masks, metric reference, and runtime. Claims are exclusive and retained
if a run is interrupted; do not silently rerun or overwrite.
