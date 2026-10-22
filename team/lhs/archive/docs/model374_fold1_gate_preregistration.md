# Model374 fold-1 continuation gates (preregistered before OOF result)

> 한국어 안내: 이 파일은 OOF 결과를 보기 전에 고정한 gate 원문을 보존하기 위해 영어로 유지합니다. 최신 결과와 현재 판정은 [한국어 최신 현황](current_status_ko.md)과 [모델 결과 기록](model_results.md)을 참고하십시오.

Purpose: limit wasted grouped-OOF training without selecting a winning blend from
one fold. Fold 1 training is fixed at 70 epochs, `val_every=0`; its held-out
predictions are not used to change training, loss, model, or blend weight.

## Gate A — standalone futility check

- Compare Model374 fold-1 image-wise mIoU (four class IoUs per image, empty
  union = 0) against Model329 probabilities on the exact same 18 held-out images.
- Model329 values come from its exactly-once specimen-group `train_oof_probs.npz`
  and are recomputed from the fold-1 masks; no official validation data is used.
- Pin the Model329 archive by SHA-256:
  `539bc1983bffb3ff55303f6865fc9a2094edc5dea9910551a481005fc1a01837`.
- Continue only if Model374 is no more than 2.0 percentage points below Model329.

## Diagnostic B — fixed half-blend (not an early-stop gate)

- On the same 18 images, compare Model329 alone with exactly one fixed blend:
  `0.5 * Model329 probabilities + 0.5 * Model374 probabilities`, followed by
  argmax and the same official image-wise empty=0 metric.
- Model329's B4 archive is already in canonical/team order
  `[Al, Al3Ni, Primary_Si, Eutectic_Si]`, matching Model374. Keep channel
  indices `[0, 1, 2, 3]`; do not permute either archive.
- No weights, thresholds, or class-specific settings are searched.
- Record its per-image and mean paired delta, but do not use it to stop or
  continue training. The 0.5 weight is merely one diagnostic setting, not a
  presumed optimum and not evidence that the blend improves.

## Blend-weight evaluation after all four OOF folds

- Predeclared scalar gamma grid: `{0, 0.1, 0.2, ..., 0.9, 1.0}`, where
  `P = (1-gamma) * P329 + gamma * P374` after channel-order alignment.
- Select on the complete grouped train-OOF set using the official image-wise
  mean of four class IoUs (empty union = 0). Let `gamma_best` be the raw maximum;
  among settings within `0.10 pp` of that maximum, choose the gamma closest to
  `0.5`, then the smaller gamma. This predeclared shrinkage avoids chasing tiny
  OOF differences without assuming that `0.5` is optimal.
- Report the full gamma curve and the selected gamma. The score at the selected
  gamma is selection-biased and is not an unbiased generalization estimate.
- A group-held-out gamma search over these four-fold OOF maps is only
  exploratory: inner-group predictions may come from fold models trained on the
  outer group, so this is not leakage-free nested CV. Fold 1 is also subject to
  the standalone continuation gate. Do not label these estimates unbiased.
- A truly leakage-free estimate of the *weight-selection procedure* would
  require nested model refits that exclude each outer group from all inner
  models; that extra training is not included in this run.
- After fixing one gamma by the preregistered rule, evaluate that single blend
  on official validation once. Because validation has been used in prior model
  searches, report it as an observed validation score, not an independent or
  confirmatory test estimate. Do not retune from that score.

## Bias accounting and later analysis

- Since fold 1 determines whether more compute is spent, report OOF scores and
  fold 2–4 sensitivity with that condition explicit. Do not call either an
  unbiased nested estimate. Do not tune any later rule using the gate outcome.
- The final gamma is selected from the predeclared train-OOF rule; validation
  labels remain unopened until that one fixed candidate is evaluated once.
- If standalone Gate A fails, stop Model374 and do not alter thresholds or
  retry a variant on fold 1. The fixed half-blend diagnostic is not a gate.

## Implementation correction record

After the first evaluation, an incorrect channel permutation `[2, 0, 1, 3]`
was found in the comparison code. Model329's own runner passes
`--candidate-order team`, whose order is `[Al, Al3Ni, Primary_Si, Eutectic_Si]`;
its saved grouped-OOF probabilities therefore already match Model374. An
independent 24-channel-permutation audit on the exact 18 fold-1 images yielded
0.769345 for identity (the highest of all 24 permutations) and 0.147515 for
the erroneous permutation. The archived Model329 fold-1 `scores.json` reports
77.9351% under its legacy skip-empty metric, so it is not directly comparable;
the corrected audit recomputes empty=0 from the exact probability maps and
masks. The candidate score is 0.764347. The original outputs are retained under their original
names but are invalid and must not be used. Corrected reruns write
`fold1_kill_switch_corrected.json` and
`fold1_futility_gate_corrected.json`; the latter is the fixed half-blend
diagnostic, not Gate A. This correction changes only channel alignment, not
training, inference, or thresholds.

The executable implementation is `experiments/model374_fold1_futility_gate.py`;
its corrected half-blend diagnostic is written to
`outputs/model374_hrnet_group_oof/fold1_futility_gate_corrected.json`.

## Corrected fold-1 paired audit

The CPU-only audit verifies that Model329 and Model374 fold-1 archives contain
the exact same 18 held-out images and that specimen groups do not cross folds.
It uses the pinned Model329 exactly-once grouped-OOF archive and saves the
per-image comparison to
`outputs/model374_hrnet_group_oof/fold1_pair_audit.json`.
The stricter post-review rerun, including fold-map parity and training-provenance
checks, is archived at
`results/outputs/model374_hrnet_group_oof/fold1_pair_audit_integrity_reviewed.json`.
The earlier file named `fold1_pair_audit_provenance_verified.json` is
superseded: it predates the runner-hash warning and stronger checkpoint and
training-log checks.

| Fold-1 OOF result (empty=0) | Image-wise mean |
|---|---:|
| Model329 reference | 76.9345% |
| Model374 standalone, evaluator FP32 summary | 76.4347% |
| Model374 standalone, saved FP16 probability archive | 76.4349% |
| Fixed 50:50 probability blend, saved FP16 maps | 77.4589% (+0.5244 pp vs Model329) |

The paired specimen-group bootstrap 95% CI is `[-0.1656, +1.1530] pp` for the
fixed blend and `[-2.1987, +1.0156] pp` for Model374 standalone. The blend is
positive on 12/18 images, but this single fold is too small to establish a
reliable gain. Mean per-class blend delta vs Model329 is `Al +0.096 pp`,
`Al3Ni -0.221 pp`, `Primary_Si +0.619 pp`, and `Eutectic_Si +1.603 pp`.
An image-level best-model oracle reaches 77.9499%; a pixel-level oracle reaches
81.7896%. Both are GT-informed, non-deployable diagnostics; they suggest that
useful complementarity is pixel/region-level rather than simple whole-image
routing.

Claude Code recommended prioritizing absent-class false-positive analysis,
class-pair and boundary/interior error decomposition, and calibration checks
once more OOF folds are available. These remain exploratory diagnostics; no
postprocessing or blend weight has been selected from fold 1. Since Gate A
passed, folds 2–4 are being trained sequentially with the unchanged fixed
recipe. Official validation remains unopened.

The fold-1 absent-class audit is at
`outputs/model374_hrnet_group_oof/fold1_absent_audit.json`. Only one
image/class pair had a GT-absent class (`Primary_Si` in
`A57_step1_sample3`), and Model329, Model374, and the half-blend all predicted
zero pixels of that class. Thus the audit found no absent-class false-positive
pixels or oracle gain in fold 1; absent-class suppression is not supported as
a priority by this fold alone. This does not rule it out on the other folds or
official valid and must be rechecked across completed train OOF folds.

## Fold-2 incremental reading rule (recorded before fold-2 OOF inference)

At this point fold-2 training is in progress, but its held-out prediction archive
has not been generated or inspected. Keep the comparison fixed at Model329 vs
Model374 and the single 50:50 probability blend; do not tune the blend weight or
any class-specific rule from fold 1 or fold 2.

- Primary descriptive summary: pool exact paired images from folds 1 and 2 and
  compute a paired specimen-group bootstrap 95% CI for the fixed half-blend
  mIoU delta versus Model329.
- Call this only a *preliminary replicated signal* if the pooled CI lower bound
  is above zero and the fold-specific point deltas are both positive. Otherwise
  report the evidence as inconclusive or not replicated; this does not alter
  the already-fixed training recipe or stop the sequential folds 3–4.
- Report per-fold and pooled class deltas and per-image changes as diagnostics;
  they do not select weights, thresholds, or postprocessing.
- Before interpreting scores, verify both models have exactly the expected
  held-out image keys for each fold, and that Model329's corresponding
  fold-specific OOF maps were produced by a model trained without that fold's
  specimen groups. Do not substitute its merged archive alone for this check.

This interim reading is exploratory, not an independent confirmatory estimate:
fold 1 already determined whether Model374 continued, and the two-fold subset
is not nested cross-validation. The full fixed protocol and its selection-bias
caveats above remain in force. Official validation is still reserved for one
final evaluation after the OOF-based choice is frozen.

### Fold-2 split/provenance check before Model374 OOF inference

Checked on 2026-10-01 before Model374 fold-2 OOF inference: the fixed grouped
split contains 52 training images from 44 specimen groups and 18 held-out images
from 15 groups, with no group overlap. Model329 fold 2 records
`train_splits=train_fold2`; its `fold2_oof/probabilities.npz` has exactly the
18 expected held-out image keys. Model374's fold-2 data links resolve to the
same `train_fold2` and `oof_fold2` split directories. The 18-image count is
therefore the grouped train-OOF fold size, not the 20-image official validation
set.

### Audit review correction and generic-ensemble control

Claude Code review found that the first version of the paired audit checked
Model329's fold-specific archive keys but scored probabilities from the merged
OOF archive. Before fold-2 Model374 predictions were available, a direct
array-by-array comparison confirmed that the 18 fold-1 and 18 fold-2 maps in
the pinned merged archive exactly equal their corresponding fold-specific
archives (36/36 exact, maximum absolute difference 0). The audit code now
scores from each fold-specific archive, asserts exact equality with the merged
archive, and records the number of parity checks. This guards against a stale
or misassembled merged OOF file. It also checks train/held-out specimen-group
disjointness, Model329's fold-specific training split/data root, and Model374's
fold manifest, epoch-70 provenance, held-out paths, and training-args hash.

Claude Code also noted that the observed blend gain could be generic ensemble
variance reduction, not Model374-specific complementarity. If and only if the
predeclared folds-1+2 preliminary-signal rule above is met, after the currently
scheduled Model374 folds finish, train a same-recipe Model329 second-seed
grouped OOF control for folds 1 and 2 (seed 1, unchanged train/held-out groups).
Compare the fixed 50:50 Model329-seed0 + Model374 blend against the fixed 50:50
Model329-seed0 + Model329-seed1 blend on the exact same images. The primary
control contrast is the paired specimen-group-bootstrap CI for the difference
between those two blend mIoUs; call Model374-specific complementarity supported
only if its pooled 95% CI lower bound is above zero and both fold-specific
point deltas are positive. Otherwise report no distinguishable specific gain.
This is an exploratory, conditional follow-up (the fold-1 continuation gate
still applies), uses no official validation data, and does not tune blend
weights. It is not launched concurrently with the fixed Model374 run.

## Fixed max-confidence gate (recorded before fold-2 OOF inference)

Claude Code proposed a learned pixel-wise stacker, then correctly withdrew its
leave-one-fold-out evaluation as invalid: the other folds' base predictors were
trained on the nominal outer fold. A genuinely nested estimate would require
many extra base-model fits and is not justified for this score-improvement
goal. Do not report the proposed leave-one-fold-out stacker score as held-out
performance.

Instead, test one parameter-free diagnostic on the saved grouped OOF maps:
for each pixel, select Model374's class prediction only when its maximum
softmax probability is strictly greater than Model329's; ties select Model329.
No temperature, confidence threshold, calibration, or class-specific rule is
fit. Use the same four-class image-wise mIoU with empty union=0. The audit
reports the rule per fold, its candidate-pixel fraction, class deltas, and
paired specimen-group bootstrap intervals against Model329 and the fixed 50:50
blend.

Fold 1 motivated this hypothesis and is exploratory only for this gate. Treat
folds 2–4 as the directional replication set, while retaining the known
fold-1 continuation-selection caveat. The gate is only worth carrying to the
single frozen official-valid evaluation if its pooled folds-2–4 delta versus
the 50:50 blend has a group-bootstrap 95% CI lower bound above zero, its point
delta is at least +0.3 pp, and each of folds 2, 3, and 4 is positive. Otherwise
discard it; do not search confidence cutoffs or alternative gate formulas.
Even if it passes, the OOF evidence is exploratory rather than independent
confirmation. The final official-valid result is reported overall and by the
frozen seen/unseen filename strata, without retuning.

## Audit review and remaining fold-1 provenance caveat

The final audit verifier additionally requires Model329's `no_valid_eval=true`,
an empty validation-loader log header, no validation-evaluation messages, and
the fixed 16,000-iteration recipe, ruling out validation-based checkpoint
selection in that run. The Model329 source currently in the project builds a
fresh model with an ImageNet encoder and no external checkpoint load; its source
hashes are captured now, not from training time. For Model374, the audit checks
the epoch-70 manifest, exact training-argument schema, no external
initialization checkpoint, same-fold paths, and checkpoint/model/evaluator
hashes against the inference manifest.

On fold 1, those model/data hashes match, but the runner SHA in the inference
manifest differs from the current runner file. The exact old runner bytes were
not retained. Also, Model329 has no per-fold inference manifest linking its
`last.pt` to the saved probability archive, and the Model374 inference manifest
does not contain the probability-archive SHA. The audit records the archive
hashes at audit time but cannot bind them retroactively to inference. It marks
provenance `partial_with_warnings`; the foldwise mIoU is recomputed from the
saved probability arrays and train-OOF masks, not regenerated from checkpoints.
Do not describe the fold-1 score as an independently rerun inference result.

## OOF versus official-valid sampling regime

A separate metadata-only check read filenames under the 70-image grouped-CV
pool and the 20-image official-valid image directory; no valid masks or pixels
were opened. There are no exact image-stem duplicates, but 7 filename-derived
specimen groups are shared. Those groups account for 7/20 official-valid images
(35%); the other 13 valid images come from groups absent from the train pool.
The group parser is the leading `A<digits>` prefix and is only a specimen-identity
proxy. Counts and group IDs are archived in
`results/outputs/model374_hrnet_group_oof/official_valid_split_metadata.json`.

Thus grouped OOF remains a valid paired comparison for the *unseen-specimen*
regime, but it is not an exact simulation of official valid's mixed regime.
At the single planned final official-valid evaluation, report the overall score
and descriptive seen-group/unseen-group strata using this frozen filename
metadata. Do not use the strata or valid labels to retune the model, blend, or
postprocessing. With only 7 seen-group and 13 unseen-group valid images, these
strata are descriptive only and their difference is not a reliable inferential
comparison. This documents an external-validity limitation; it does not
invalidate the paired OOF comparison or by itself prove label leakage.
