# Claude Code review: post-Model400 train-OOF intervention selection

This is a read-only scientific review. Do not edit files, launch commands, train,
or access official validation/test data. Read only the referenced local
protocols/results. Return exactly one overall recommendation:
(A) PRIORITIZE A DIRECTION-TARGET HEAD PILOT FOR Al3Ni/Primary Si,
(B) PRIORITIZE A STEP4-CONDITIONED Eutectic-Si PILOT,
(C) PRIORITIZE THE EXISTING Model399 WEIGHTED-CE DRAFT, or
(D) DEFER/NO-GO. Choose one candidate only; do not propose a sweep. Give the
smallest discriminating train-only grouped-OOF design and explicit stop rules.
This is advice, not authorization to edit, train, or access official
validation/test.

## Current target and constraints

- Official valid leader remains Model364 at 79.9429727% image-wise four-class
  mIoU (empty union=0), about 0.0570 percentage points below 80%. This is a
  repeatedly reused 20-image development set, not independent test evidence.
- Model392 (HRNet-W48/U-Net++ width-normalized) scored 79.0437% on its
  two-seed × four-fold train-only grouped OOF versus Model382 78.9395%;
  its preregistered adoption gate failed, and its official-valid evaluation
  was not authorized.
- The fixed Model397 B4/HRNet blend was 79.9859% train-OOF, not an official
  validation score. Do not interpret it as 80% validation performance.
- No test labels/scores are available. Avoid any test-time tuning, validation
  search, threshold tuning, or route search.

## Completed Model400 official-valid measurement

After an independent review, one fixed Model396-D2 route was evaluated once:
`0.5 * Model329 B4 four-fold probabilities + 0.5 * Model382 seed2 four-fold
probabilities`, with argmax after averaging. HRNet used patch 512, input scale
2, overlap 170, flip4 TTA and Gaussian tile stitching. On the 20 official-valid
images it scored **79.3227251%** with image-wise four-class mIoU, empty=0;
this is 0.6202476 pp below Model364 and does not meet 80%. Class image-means
were Al 94.4481%, Al3Ni 74.8952%, Primary Si 86.7512%, Eutectic Si 61.1964%;
A71 was lowest at 39.700%. No test files were accessed. This is
selection-conditional evidence because valid has been reused, and the
four-fold valid inference route differs from Model396's one-held-out-fold OOF
estimand. Do not propose another valid evaluation or tune this route.

The score is recorded in `experiments/results/model400_model396_d2_valid_once.json`;
the fixed recipe and caveats are in
`experiments/model400_model396_d2_valid_ensemble_preregistration.md`.

## Existing fixed candidate with a passing OOF screen

Model396 is a fixed 1:1 probability blend of Model329 B4 and one Model382
HRNet seed. Both predeclared arms passed the paired grouped-OOF screen. D2
(B4 + seed2) scored 79.9866% train-OOF, with D2−two-HRNet-seed mean B2
+0.8244 pp (95% group-bootstrap CI +0.4339 to +1.2615; all four folds
positive); all four classes and Step4 improved. D2−its single HRNet seed was
+1.0266 pp (CI +0.5832 to +1.5352). These are selection-conditional OOF
results, not official-valid results. The Model396 protocol only permits
considering a separately locked final candidate; it contains no valid-stage
authorization. See:

- experiments/model396_fixed_architecture_oof_preregistration.md
- experiments/results/model396_fixed_architecture_oof_lock.json
- experiments/results/model396_fixed_architecture_oof.json

On P100, only the four fold checkpoints per family are present; there is no
full-train Model329 or Model382 checkpoint. A possible non-training valid
candidate would average four Model329 fold predictions and four Model382
seed2 fold predictions per image, then use the fixed 1:1 family mean. The B4
valid probability ensemble already exists; the HRNet seed2 valid maps do not.
This changes valid inference from the OOF arm's one-fold-per-image prediction
to a four-fold ensemble, so it is not an exact reproduction of D2's OOF
estimand. Assess whether that is a defensible, useful one-shot measurement or
whether a different candidate is needed. Do not score it during this review.

## New train-only evidence

Model392 already contains a class-agnostic one-channel boundary auxiliary head
trained with boundary BCE+Dice. It was not used for semantic inference.
We measured its sigmoid localization on all 70 held-out train images (59
specimen groups), each predicted by both held-out-fold seeds: 140 image-seed
records, 8 checkpoints, full image coverage. Fixed flip4 TTA and Gaussian
stitching were used with preregistered 25% tile overlap (less than the original
semantic OOF overlap for diagnostic efficiency). Group-macro per-image AP was
92.0032% (95% specimen-group bootstrap CI 91.1800–92.7993%) versus 30.2709%
positive-pixel prevalence; paired AP-minus-prevalence was +61.7323 pp (CI
[+59.0827,+64.2441]). Seed means were 92.0292% and 91.9772%; fold means were
90.8125–93.2972%. The frozen rule returned LOCALIZATION_SIGNAL.

This is selection-conditional train-OOF diagnostic evidence only. It measures
class-agnostic annotation-boundary localization, not class-side direction
accuracy, corrected semantic predictions, or mIoU gain. The 25% overlap also
differs from Model392's denser semantic OOF inference. Full protocol, code,
lock, and JSON are:

- experiments/model392_boundary_head_localization_preregistration.md
- experiments/model392_boundary_head_localization.py
- experiments/results/model392_boundary_head_localization_lock.json
- experiments/results/model392_boundary_head_localization_oof.json

Earlier Model387 error geometry found Al3Ni and Primary Si false negatives
enriched within 4 px of GT boundaries, while most Eutectic Si errors were in
the deep interior; consult the relevant section of model_results.md. Therefore
a class-agnostic correction could be harmful, and boundary work alone will not
address the Eutectic interior failure mode.

## Completed mask-only direction-target support audit

A preregistered CPU audit used only the outer-training masks in the same four
group folds; it did not access OOF predictions or official valid/test images,
train a model, or compute mIoU. At same-class interior clearance >=4 px and
maximum source-to-target distance 16 px, supported boundary-source fractions
by fold were 71.75–75.31% for Al3Ni and 94.11–95.49% for Primary Si. All eight
compass directions occurred in every fold. Per-direction group coverage was
44/44–45/45 for Al3Ni and 43/44–44/44 for Primary Si. At a stricter 4 px
maximum offset, support fell to 37–41% and 65–68%, respectively. This
establishes target-geometry feasibility only, not direction prediction or
segmentation benefit. Protocol, code, lock, and result:

- experiments/model392_direction_target_support_preregistration.md
- experiments/model392_direction_target_support.py
- experiments/results/model392_direction_target_support_lock.json
- experiments/results/model392_direction_target_support.json

## Competing Model397 train-OOF error pattern

The fixed Model397 blend C (B4/Model392 probabilities, not a learned router)
has a separate post-selection train-OOF error-type audit. Across all 70 images,
Eutectic-Si present-image macro recall/precision was 72.96/76.18%, and its
predicted/GT area ratio was 1.023. On the 27 Step4 images (26 specimen groups,
all with Eutectic GT), recall/precision was 65.78/74.74%, while predicted area
was only 0.899 of GT; the blend still made 553,902 Eutectic FN and 379,332 FP
pixels. A global area-prior adjustment is therefore not an obvious fix. Fold
support is uneven (Step4: 9/9, 7/7, 4/4, and 7/6 images/groups across folds),
so a Step4-conditioned intervention needs cautious grouped safety checks.
The audit did not rescore mIoU, and these are selection-conditional train-OOF
diagnostics, not independent evidence. See:

- experiments/model397_train_oof_error_type_audit.py
- experiments/results/model397_train_oof_error_type_audit.json
- experiments/model397_fixed_b4_model392_blend_preregistration.md
- experiments/results/model397_fixed_b4_model392_blend_oof.json
- experiments/model399_class_weighted_hrnet_finetune_draft.md

## Questions

1. Which single train-only intervention (A–C) is most justified by the
   70-image/59-group OOF evidence? Model400's below-target score confirms the
   goal is unresolved, but its reused-valid class profile must not select or
   tune the intervention. If none has a credible discriminating design,
   choose D.
2. Given that the boundary head localizes boundaries but Model392's semantic
   OOF gate failed, is a SegFix-like direction-to-same-class-interior head a
   plausible non-routing intervention, or should it be rejected?
3. If a direction-head pilot is justified, specify a minimal architecture and target:
   whether to predict 8 spatial directions, restrict correction to Al3Ni and
   Primary Si, how to define/ignore pixels without a valid same-class interior,
   how to prevent boundary-score calibration/threshold tuning, and how raw
   semantic logits remain the fallback.
4. Specify leakage-safe training and evaluation: paired fixed specimen-group
   folds/seeds, outer-train-only target/support construction, image-wise
   four-class mIoU with empty=0, group bootstrap, and per-class/Step4/fold/
   per-image safety checks. Propose an OOF adoption threshold that is
   meaningful relative to observed seed/fold noise, not an arbitrary demand
   for a large gain.
5. Account for known negative/duplicate experiments in model_results.md:
   class-agnostic boundary CE, pairwise boundary corrections, directional
   loss pilots, high-resolution/scale probes, prior correction, and routing
   should not be repeated without a specific new mechanism.
6. Compare the Model400 validation class weakness only as a descriptive
   observation against train-OOF mechanisms; do not use it to choose or tune
   a branch. Given Model397's Step4 Eutectic FN and FP together, state whether
   that candidate is more justified than the Al3Ni/Primary Si direction head.
   If recommending Model399, explain how its paired controls address both
   Eutectic false negatives and false positives despite near-source crop
   priors and prior valid harm from unweighted logit correction. Distinguish
   boundary-localized Al3Ni/Primary Si errors from Step4 interior errors.

Do not claim either the AP or support audit demonstrates mIoU improvement.
Do not authorize further official-valid access in this review. Any training
still requires a separate preregistration, source/inputs lock, and explicit
human review of your recommendation.
