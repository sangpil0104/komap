# Model410 — small-component-balanced crop sampling

Status: pre-lock revision 1.2, independently reviewed. It replaces the
unexecuted weighted-CE draft. No
Model409 partial or final metric may be read or used to select this treatment.
The Model409 exposure addendum remains in force. No official valid/test data
are in scope; any train-OOF result is development evidence, not confirmation.

Revision note (2026-10-06, before source lock or optimizer steps): a train-only
metadata audit found original image widths of 675, 1230, 1278, and 1440 pixels,
while the trainer rescales crop size to each fold's modal width. Fixed raw-pixel
area bins would therefore represent materially different apparent sizes. The
earlier raw-native-bin draft is superseded and was not executed. This revision
defines bins in modal-width-equivalent area; it uses no held-out labels or
official valid/test data. All source images are larger than their training
crops, so reflection padding does not alter the full-image component map used
for anchor selection.

Revision 1.2 note (still before source lock or optimizer steps): the inherited
small-component diagnostics use raw native-pixel areas, which do not align with
the width-normalized sampler bins. Preserve those legacy diagnostics and their
existing safety gate, and add a width-normalized per-bin hit/false-island report
to the one-shot OOF scorer. This added report is descriptive only; it does not
change the primary mIoU or any gate.

## Independent review and hypothesis

Do not continue the route/ensemble search here. Model404/406/408 show that
raising small-component recall through an auxiliary loss can also create many
unmatched tiny predictions; Model407's direct FP penalty traded away true
small-component recall. Model409 tests a distinct local-feature objective but
has not been scored. A global inverse-frequency CE loss is not a direct
small-object intervention, and earlier class weighting has shown boundary
bleed risk. The crop-prior audit also found the rare-centered pixel sampler's
class pixel prior close to the whole-mask prior, weakening the case for a
global prior correction.

The remaining, falsifiable sampling hypothesis is that selecting rare-phase
anchor pixels uniformly makes a connected component's chance of being chosen
roughly proportional to its area. Thus, at equal class/epoch crop budgets,
small isolated components may receive fewer centered training views than
large regions. Test only a small-component-balanced sampler; leave the model,
loss, optimizer, augmentations, inference, and checkpoint initialization
unchanged. It adds no route, ensemble, threshold, class weight, or inference
postprocessing. This is a transfer experiment, not a claim that every tiny
annotation is meaningful.

## Frozen arms and recipe

- Four fixed specimen-group folds, exactly-once held-out train predictions
  for 70 images / 59 groups. Both arms start from the same Model392 seed-1
  HRNet-W48/U-Net++ checkpoint for that fold and train for eight epochs.
- Control: unchanged `KomapPatchDataset` sampling (50% positive crop anchors
  sampled from the existing Al3Ni/Eutectic-Si pixel coordinates, 50% uniform
  crops).
- Candidate: retain the same 50/50 positive/uniform probability. For each
  image, convert a component's raw area `A` to the fold-modal-width coordinate
  `A_norm = A * (target_width / image_width)^2`. For a positive draw on an
  image containing eligible components, choose uniformly among available
  target classes, then uniformly among nonempty `A_norm` bins `[1,4)`, `[4,16)`,
  `[16,64)` pixels, then uniformly among 8-connected components in that
  class/bin, and finally uniformly among that component's pixels. Keep the
  current crop-center jitter (`crop_size/4`) and boundary clamping. Components
  with `A_norm >= 64` are not anchors. If the image has no eligible component,
  use the unchanged rare-pixel positive sampler; if it has no target pixels,
  use a uniform crop.
- Paired stream: same parent, seed, 12 crops/image/epoch, batch 4 × gradient
  accumulation 2, 512 patch, width-normalized scale 2, per-item deterministic
  crop/augmentation RNG, CE+Dice plus the existing boundary loss, AdamW,
  encoder LR `3e-5`, decoder LR `1e-4`, weight decay `1e-4`, eight-epoch cosine
  schedule, AMP, and inference/TTA. Only the crop anchor-selection rule
  differs. No validation split is loaded during training.
- Before source locking, a mask-only preflight verifies all normalized bins
  have support in each outer-training fold and checks that the full-image
  component maps need no reflection padding. The preflight reads no held-out
  or official valid/test labels. The Model392 trainer is invoked unchanged through a narrow dataset-class
  injection wrapper. Parent checkpoint preprocessing metadata must match
  patch 512, input scale 2, width normalization, and each fold's modal width.
  The sampler-arm receipt records the exact area-coordinate formula and bins.
  Each arm/fold has a new no-overwrite output path and a sampler-arm receipt.

## Evaluation and stop rule

Primary metric: official image-wise mean mIoU on held-out train groups: four
class IoUs per image, empty union = 0, average classes then images. Compare
candidate with its newly trained, compute-matched control using 20,000 paired
specimen-group bootstrap draws stratified by the four fixed folds. Also report
the fixed Model392 parent as a safeguard, all four class deltas, Step4 delta,
fold deltas, legacy native 1–63px GT component hit rates and unmatched
same-class predicted components/FP pixels, plus descriptive GT-hit and
unmatched-prediction counts in the matching modal-width-equivalent `[1,4)`,
`[4,16)`, `[16,64)` bins. The width-normalized diagnostic is not an added gate.

Seed-1 screen passes only if all hold: candidate-control gain at least `+0.20
pp`, paired group-bootstrap 95% CI lower bound `>0`, at least 3/4 folds
positive, worst fold `>=−0.30 pp`, no class or Step4 regression below
`−0.30 pp`, candidate is not below the Model392 parent point estimate, tiny
GT component hit-rate does not fall for either target class, and unmatched
1–63px predicted components do not increase for either target class. A
failure is NO-GO; do not tune or rerun on these same folds. A pass only
permits a separately source-locked seed-2 train-OOF confirmation. No official
valid look is authorized by this protocol and test remains sealed.

## Implementation and integrity checks

Before any optimizer step: verify all fold/group manifests and parent-checkpoint
hashes; run the synthetic component-bin/determinism smoke and the width-normalized
scorer self-test; confirm the control path calls the exact base sampler; compare
forwarded training args and parent preprocessing metadata; lock all source hashes;
ensure no-overwrite paired outputs; and test the prediction-only/scoring path on
synthetic maps. The
held-out predictor opens images only and writes probability maps—never masks,
per-image scores, or fold summaries. The one-shot final scorer alone may first
read held-out train masks and emit aggregate results after all eight arm/fold
fits and all 70 predictions finish.

## Explicitly rejected draft

`model410_weighted_ce_preregistration_draft.md` is superseded, not executed.
Its inverse-frequency CE plus probability correction addresses global class
prior, not connected-component exposure; weighted CE + Dice/boundary does not
justify an exact posterior correction, and the large Eutectic-Si weight risks
repeating prior overprediction. This replacement is a single predeclared
sampler comparison, not a sweep. The two protocols must never both be run as
Model410.
