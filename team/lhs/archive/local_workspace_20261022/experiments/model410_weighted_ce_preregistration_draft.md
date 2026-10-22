# Model410 draft: weighted-CE HRNet with matched probability correction

> **Superseded before execution.** Independent review replaced this global
> class-prior experiment with the single-model small-component-balanced crop
> sampler in [the Model410 preregistration](model410_small_component_sampler_preregistration.md).
> This draft must not be trained or scored as Model410.

Status: preparation only. Model409 currently owns the P100 run; do not start
this experiment until its runner terminates and its locked OOF result is
scored. Claude Code is unavailable, so this is an independent review, not a
Claude-reviewed or locked protocol. No official valid/test data are in scope.

## Question and rationale

Does the teammate-reported pairing of inverse-training-frequency CE weights
and the matching probability correction improve a single HRNet-W48/U-Net++
model on the fixed specimen-group train OOF, relative to an otherwise matched
unweighted continuation? This is a standalone training/loss experiment: no
routing, ensemble, threshold search, or class-specific inference branch.

The teammate report describes HRNet-W48/U-Net++, 512px patches, input scale
2, CE weights inverse to training-mask pixel frequencies, and division of
post-TTA/post-overlap probabilities by those same weights before argmax. It
reports 81.55% on 20 reused valid images, but the recipe has not been
independently reproduced and the valid score is selection-prone. The existing
Model399 CPU audit found the actual rare-centered crop prior close to the
outer-train pixel prior; it does not show a need for a large sampler
correction. The existing loss source uses PyTorch weighted-mean CE plus
class-mean Dice and a boundary loss. Consequently, `p / w` is an empirical
transfer of the reported correction, not a guaranteed posterior identity:
Dice/boundary terms and batch-normalized weighted CE violate the simple
label-shift derivation. PyTorch's weighted-mean CE divides each minibatch's
weighted loss sum by the sum of the target weights in that minibatch, so the
effective batch scale varies with crop composition. This experiment tests the
complete fixed recipe, not the theory in isolation.

## Frozen candidate and control

- Four existing specimen-group folds, exactly-once held-out train predictions
  for all 70 train images / 59 groups. Use the fixed Model392 seed-1
  fold-specific HRNet-W48/U-Net++ checkpoints as common initializations.
- Candidate `Wcorr`: eight-epoch continuation with CE weights
  `w_c = (1 / p_c) / mean_j(1 / p_j)`, where `p_c` is computed from pooled
  pixel counts in that fold's outer-training masks only; retain the existing
  Dice and boundary losses. At inference, apply the existing TTA and tile
  probability averaging first, then divide each class probability by its
  same fold-local `w_c` and take argmax. Renormalizing after division is
  unnecessary for argmax and is not a tuning parameter.
- Matched control `U`: same checkpoint, eight epochs, optimizer, schedule,
  crop/RNG stream, augmentation, batch, and inference, but unweighted CE and
  no probability correction. Change only the CE weight vector and the
  correction that is inseparable from the candidate recipe.
- Also record the candidate's uncorrected output as a prespecified mechanism
  diagnostic, not a selectable arm. Do not choose between corrected and raw
  output after inspecting OOF results; corrected output is the only primary
  candidate.
- Match the Model404 continuation recipe for all other training settings.
  Before source lock, independently verify its exact CLI and checkpoint
  preprocessing metadata. Fail closed if the fold's target width/input scale
  do not exactly match its parent checkpoint. Use the existing trainer only
  if its source hash and class-weight behavior match this protocol.

## Evaluation and decision rule

Use official image-wise mIoU on held-out train folds: four class IoUs per
image, empty union = 0, average classes then images. Compare `Wcorr` against
`U` with 20,000 paired specimen-group bootstrap draws stratified by fold.
Report paired delta, CI, fold deltas, all four class deltas, Step4, per-image
deltas, Al3Ni/Eutectic-Si predicted area and false-positive burden, and
small-component hit-rate/unmatched-component counts. The primary score is not
pooled-confusion mIoU.

Seed-1 screen passes only if every condition holds: gain at least +0.20 pp,
paired group-bootstrap 95% CI lower bound > 0, at least 3/4 folds positive,
worst fold no worse than -0.30 pp, no class or Step4 delta below -0.30 pp,
and neither target class increases unmatched 1–63 native-pixel predicted
components. A pass permits a separately locked seed-2 confirmation only; a
failure is NO-GO and does not trigger official valid/test evaluation. Seed-2
must have positive paired gain and pass the same safety conditions before
any later one-shot valid decision is considered. No result here estimates
test performance.

## Independent review notes / risks

- The 81.55% teammate score is on a reused 20-image validation set; it is a
  hypothesis, not an expected result. The separate B4 code archive does not
  reproduce the HRNet loss recipe.
- Previous inverse-frequency weighting without a matched correction caused
  Eutectic-Si overprediction/edge bleed; a prior-only correction on an
  unweighted model was also harmful. The paired treatment is therefore
  essential, and tiny false-positive diagnostics are adoption guards.
- `p_c` must be derived independently within each outer-training fold. Never
  use held-out masks, valid masks, or target information. Scale all four
  weights to arithmetic mean one and save exact counts and values in the
  immutable lock.
- Model399's older draft centered its final adoption estimand on an ensemble
  and required Claude review. This draft supersedes that framing: no blend or
  routing is involved. It remains unreviewed by Claude and must be independently
  source/data locked before training.
- The short continuation is a transfer test of the reported loss/correction,
  not an exact reproduction of the teammate's 70-epoch training from
  ImageNet. Do not label it an exact reproduction.
- Primary-source check: [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
  specifies the weighted-mean denominator; [Long-tail learning via logit
  adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)
  motivates prior-based logit adjustment in long-tail classification, while
  [ICML label-shift correction](https://proceedings.mlr.press/v235/wei24g.html)
  depends on label-shift assumptions. Neither establishes this post-TTA
  correction for KoMaP segmentation or its image-wise mIoU.

## Required preflight before any optimizer step

Verify fixed split/group manifests, outer-train-only class counts, parent
checkpoint hashes and preprocessing metadata, trainer/loss/evaluator hashes,
candidate/control identical RNG and optimizer settings, no-overwrite output
paths, and a synthetic test proving that the correction divides averaged
probabilities by the exact fold-local weights before argmax. Lock all of these
inputs and the one-shot OOF scorer. Do not inspect Model409 OOF per-image
values while preparing this protocol; Model409's aggregate gate may determine
whether this follow-up is worth running, but cannot change Model410's recipe.
