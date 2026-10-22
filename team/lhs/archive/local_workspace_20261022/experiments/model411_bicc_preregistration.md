# Model411 — component-balanced bidirectional small-island loss

Protocol: conditional Model411 candidate only after a Model410 NO-GO and a
free shared P100 lock. The recipe and one-shot rules below are frozen; any
training requires successful source locking and outer-train-only calibration.
Current progress and implementation history are recorded in `model_results.md`.

Pre-lock implementation correction (2026-10-06): the first conditional queue
attempt completed the synthetic loss smoke and fold-manifest checks, then
stopped before calibration or any optimizer step because the Model411 source
lock referenced a nonexistent `CALIBRATION_LOCK` attribute on the Model406 v3
lock module. Model406 v3 inherits that immutable calibration reference from
its v2 lock module. The implementation now references and hashes that exact
v2 lock file. This is a provenance-tool correction only; the Model411 loss,
weights, training recipe, folds, controls, and score gates are unchanged. No
Model411 source/calibration/OOF lock, checkpoint, prediction, or result was
created by the failed preflight.

## Why this is not another route or ensemble

The fixed 70-image/59-group train-OOF results show a recurring trade-off:

- Model404's GT-component recall term improved tiny Al3Ni/Eutectic-Si hit rates
  by about 1.7–1.8 pp, but image-wise mIoU moved only +0.032 pp, below the
  +0.20 pp screen threshold.
- Model406's recall plus pixel hard-negative term moved mIoU +0.020 pp and
  increased unmatched tiny predictions by about 18 components/image for each
  target class.
- Model408's targeted confuser hinge moved mIoU +0.029 pp, but likewise
  increased unmatched tiny predictions and did not pass the screen.
- Model407's prediction-component penalty reduced tiny false positives, but
  reduced tiny GT-component hit rates by 2.1–3.9 pp and mIoU by 0.087 pp.

Independent reading suggests a concrete implementation issue worth testing,
not a reason to assume a gain: Model407 sums false-component penalties within
each crop/class cell, so crops with more islands receive proportionally more
suppression pressure. Model411 instead gives each eligible small component one
term and averages components within crop×class×size-bin cells. It pairs that
precision-side term with a fixed, smaller GT-recall branch. This tests whether
component-normalized bidirectional supervision avoids both the positive-only
island growth and the count-sensitive over-suppression seen earlier.

The idea is informed by BiCC, a 2026 preprint that combines annotation-derived
and prediction-derived component partitions and reports a precision/recall
trade-off on five 3D medical segmentation datasets. Its task, dimensionality,
and lesion-level metrics differ from KoMaP's 2D metallography and pixelwise
image-mean mIoU; it is a hypothesis source, not evidence of expected KoMaP
improvement: https://arxiv.org/abs/2609.30223.

## Frozen treatment

- Parent: the existing Model392 seed-1 HRNet-W48/U-Net++ checkpoint for each
  of the four fixed specimen-group folds. Candidate fine-tunes for exactly
  eight epochs with patch 512, width-normalized input scale 2, the existing
  per-item crop/augmentation RNG, AdamW, and the existing CE+Dice+boundary
  objective.
- Primary control: the already locked Model406 v3 matched continuation-control
  probabilities. This is the same Model392 parent and eight-epoch CE+Dice+
  boundary recipe; do not spend a second set of four folds retraining it.
- Target phases are fixed to Al3Ni and Eutectic Si. Find 8-connected components
  on the post-augmentation 512×512 training mask/logit grid. Use fixed area
  bins `[4,16)`, `[16,64)`, `[64,256)` augmented pixels as the existing nominal
  native 1–63px proxy. Exclude crop-border components.
- GT branch: for each eligible GT component, use
  `-log(mean p(true class))`; average components within each crop×class×size
  cell, then average active cells equally.
- Prediction branch: use detached four-class argmax and 8-connectivity. Keep
  eligible target-class components in the same fixed size bins only when they
  have zero overlap with a one-pixel same-class GT halo. For each, use
  `-log(1 - mean p(target class))`; average components within each
  crop×class×size cell, then average active cells equally. No component-count
  sum, probability threshold, inference filter, or post-hoc correction.
- Fix the auxiliary branch mixture at `0.75 × GT + 0.25 × prediction`, before
  examining any Model411 OOF result. A single fold-specific multiplier is
  calibrated to a 10% median decoder-gradient norm ratio from 32 outer-train
  crops per fold, with zero optimizer steps. Reject unsafe or insufficient
  calibration; no coefficient/alpha sweep.
- Candidate output is a single four-class model. No routing, ensemble,
  checkpoint blending, test-time class selection, or official-validation
  access.

## Scope and one-shot evaluation

Use the fixed four-fold specimen-group split with exactly-once held-out
predictions for all 70 `train.zip` images / 59 groups. The predictor may read
held-out images only and must write probabilities without masks or fold
scores. After all four candidate folds are complete, verify immutable source,
parent, split, and OOF artifact locks, then run one scorer against the locked
Model406 v3 control and fixed Model392 parent. Metric: official image-wise mean
of the four class IoUs, with empty union set to zero; uncertainty via 20,000
paired specimen-group bootstrap draws stratified by fold. Report class and
Step4 deltas, native tiny/small GT-component hit rates, unmatched target-class
1–63px predicted components/image, and target-class FP pixel burden.

The screen passes only if all conditions hold: candidate-control gain at least
`+0.20 pp`, paired group-bootstrap 95% CI lower bound `>0`, at least 3/4 folds
positive, worst fold `>=−0.30 pp`, Step4 and each class delta `>=−0.30 pp`,
candidate not below the Model392 parent point estimate, tiny GT hit-rate delta
for each target class `>=−0.30 pp`, and unmatched 1–63px predicted components
and target-class FP pixels do not increase for either target class. Failure is
NO-GO: do not tune this recipe on the same OOF folds and do not open official
valid/test. A pass permits only a separately source-locked seed-2 train-OOF
confirmation.

## Implementation readiness and stop conditions

The loss, narrow wrapper over the unchanged Model404 trainer, outer-train-only
zero-step calibrator, image-only predictor, source/calibration/OOF lock tool,
runner, one-shot scorer, and Model410-gated queue are implemented. P100
`komap` py_compile, shell syntax, CLI import/help, and CPU synthetic/dense
component smokes pass. Before any optimizer step, still required: verify the
Model406-control and Model392-parent artifact hashes by creating the source
lock, run the 32-crop/fold zero-step calibration, and measure CPU component
labeling overhead without GPU allocation. The queue verifies Model410's
result/source/OOF/claim chain and obtains the exclusive P100 lock only after
its one-shot screen completes. The Model411 source lock records that no-go
evidence. If Model410 passes its predeclared gate, pause Model411 for a fresh
review rather than automatically spending another training run. OOF lock and
scoring occur only after all four candidate folds and held-out predictions
are complete.

Neither a train-OOF screen nor this preregistration is an official-valid or
test score. No score gain is predicted.
