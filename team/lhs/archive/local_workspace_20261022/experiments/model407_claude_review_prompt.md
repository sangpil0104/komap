# Claude Code read-only review — Model407 small-component correction

Submit once after the complete Model407 OOF scorer, tiny-component audit, and
separately preregistered FP-area audit have finished and their locks verify.
The current CLI quota reset is scheduled for 2026-10-06 23:00 KST; do not retry
or route around quota before then. This is a read-only review: no edits,
training, official-validation/test access, threshold/blend/route search, or GPU
utilization questions.

```text
Act as an independent segmentation researcher. Review the completed and
integrity-verified Model407 train-only grouped-OOF experiment and its
descriptive predicted-component FP-area audit. Read only the locked summary
artifacts and listed repository source/protocol files. Do not edit files, run
training, inspect official validation/test images, masks, predictions or scores,
search thresholds/blends/routes, or ask about GPU utilization.

Task metric: per image, compute IoU for four classes, assign IoU=0 if a
class-union is empty, average the four class IoUs, then average images. The
same 20-image official validation set has been repeatedly used; its raw leader
is selection-conditioned and must not be accessed for this review. Test stays
sealed. Train-OOF is development evidence, not an independent estimate.

Experiment question:
Model406 improved tiny Al3Ni/Eutectic-Si GT-component recall but increased
small, zero-overlap predicted islands. Model407 preserves the balanced GT
component-recall term and replaces Model406's pixel top-0.1% hard-negative
term with a detached-argmax 8-connected predicted-component FP penalty.
Predicted components touching a one-pixel same-class GT halo or crop border
are protected. Each unmatched component contributes
`-log(1 - mean(p_target))`, summed per crop/class cell. This is
count-sensitive, not an explicit native-size-weighted objective. The
outer-train-only coefficient calibration used 32 crops/fold, zero optimizer
steps, and targeted 10% median decoder-gradient norm.

Protocol: fixed 4-fold specimen-group split, 70 train images / 59 specimen
groups; per-fold fixed Model392 seed-1 initialization; 8 epochs, same recipe,
seeded data order, augmentation, optimizer, and inference as the matched
Model406 v3 control; native-resolution image-wise four-class mIoU with
empty-union=0; 20,000 paired specimen-group bootstrap draws. The score's
primary comparison is Model407 candidate vs the already-locked Model406 v3
control. Candidate vs the newly trained Model406 pixel-hard-negative arm is
descriptive and cannot rescue a failed primary gate.

Adoption requires every preregistered safeguard: primary image-wise mIoU
delta >= +0.20 percentage points; paired group-bootstrap 95% CI lower bound
> 0; >=3/4 positive folds; worst fold >= -0.30 pp; Step4 >= -0.30 pp; no
target-class, all-class, parent, or other inherited guard regression; and no
increase in unmatched native 1-63 px predicted components for either target
class. A pass authorizes only an independently locked seed-2 grouped-OOF
confirmation, not an official-validation look.

The separate component-area audit is descriptive only. It partitions
Al3Ni/Eutectic-Si FP pixels by native predicted-component area (1-63,
64-255, >=256 pixels), same-class GT overlap, and whether GT contains the
class. It neither changes the primary score/gate nor authorizes model
selection. Under empty-union=0, an absent GT class remains IoU=0 regardless
of false predictions; do not claim direct within-class IoU gain from reducing
those FPs.

Before interpreting anything, verify the complete source/calibration/OOF
locks, one-shot claim, 70-image/59-group coverage, input hashes, and that the
control artifact matches the registered provenance. If any check fails, stop
and report it rather than using partial results. Do not use per-fold
intermediate summaries to choose among treatments.

Read these artifacts after the owner has copied their small JSON summaries
from P100 and verified each SHA-256 against the remote copy:
- experiments/model407_component_fp_preregistration.md
- experiments/model407_component_fp_loss.py
- experiments/model407_component_fp_train.py
- experiments/model407_component_fp_lock.py
- experiments/score_model407_component_fp_oof.py
- experiments/model407_tiny_component_audit.py
- experiments/model407_predicted_component_error_audit_preregistration.md
- experiments/model407_predicted_component_error_audit.py
- experiments/results/model407_component_fp_oof.json and its `.claim`
- experiments/results/model407_component_fp_source_lock.json
- experiments/results/model407_component_fp_calibration_lock.json
- experiments/results/model407_component_fp_oof_lock.json
- experiments/results/model407_tiny_component_confusion_audit.json
- experiments/results/model407_predicted_component_error_audit.json
- the locked Model406 v3 result/lock and Model392 parent summary used by the
  Model407 scorer
- model_results.md sections for Models 392, 402-407
- experiments/model392_direction_target_support_preregistration.md and its
  result, only to assess whether boundary-direction correction is relevant

Return concise, evidence-grounded answers:
1. Is there any critical leakage, split, scoring, implementation, or lock
   defect that invalidates the comparison?
2. Does the candidate pass every primary and inherited gate? Report the exact
   paired mIoU delta/CI/fold consistency and class/Step4/component safeguards.
3. Which residual error mechanism is actually supported: isolated small FP
   islands, GT-overlapping/large FP regions, class confusion within tiny GT
   components, or absent-class predictions? Cite the area and class-confusion
   summaries; distinguish error counts from pixel mass and from mIoU gain.
4. Recommend at most one distinct next experiment, or recommend no-go, based
   on those residuals and prior negative experiments. Compare only relevant
   mechanisms (e.g. explicit size-aware component FP weighting, tiny-GT local
   feature separation, or boundary-to-interior direction correction); do not
   assume any is beneficial. Respect existing no-go results for global
   supervised contrastive, rare-component crop/anchor sampling, generic
   pairwise margin, and the exploratory Model385/386/387 limits. State a
   concrete grouped-OOF matched control, primary metric, critical safeguards,
   and stop/adoption rule. Do not forecast an mIoU gain.
```
