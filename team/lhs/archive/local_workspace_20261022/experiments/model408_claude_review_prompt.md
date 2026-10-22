# Claude Code read-only review — Model408 tiny-GT confusion correction

Submit only after Model408's complete 4-fold train-OOF scorer and secondary
component audit finish, all source/calibration/OOF locks verify, and the small
JSON result summaries have been copied from P100 with hashes checked. The
current Claude CLI quota reset is scheduled for 2026-10-06 23:00 KST; do not
retry or route around the quota before then. This is read-only: do not edit,
train, inspect official validation/test, search thresholds/routes/blends, or
ask about GPU utilization.

```text
Act as an independent segmentation researcher. Review only the completed,
integrity-verified Model408 grouped train-OOF artifacts and relevant source and
protocol files. Do not inspect partial per-fold scores, official validation or
test images/masks/predictions/scores, or perform training or parameter search.

Task metric: for each image, compute IoU for all four classes, assign 0 when a
class union is empty, average the four class IoUs, then average over images.
Train-OOF is development evidence, not an independent estimate. The repeatedly
used 20-image official valid set is selection-conditioned and test stays
sealed.

Project-level target context: the recorded highest official-valid result is
Model364 at 79.9429727% (20 reused validation images; not independent
confirmation). Model397's fixed B4/Model392 blend is the strongest recorded
train-OOF model at 79.9858620%, but its additional valid look was explicitly
disallowed by the Model392 gate. Model405 already replaced the Model392 seed-1
branch in that Model397 blend with an earlier small-component branch; it scored
79.9700230%, delta -0.0158390 pp, and was NO-GO. Model408 is a Model392-only
fine-tuned branch, not a new best ensemble. Therefore, any next recommendation
must explain how it would transfer to the actual best pipeline and why it is
not merely another branch substitution.

Experiment motivation and treatment:
Model407's predicted-component FP suppression lowered small predicted islands
but harmed recall of true tiny GT components, especially Al3Ni and Eutectic
Si. The locked confusion audit showed tiny Al3Ni pixels chiefly confused with
Al, and tiny Eutectic-Si pixels with Al/Al3Ni. Model408 starts from the fixed
Model392 seed-1 fold checkpoints and uses the Model404 continuation recipe.
It replaces the candidate auxiliary component loss with a zero-margin hinge
only on post-augmentation GT components in fixed area bins [4,16), [16,64),
[64,256) pixels at nominal 2x scale; fixed confusers are Al3Ni->Al and
Eutectic-Si->Al/Al3Ni. CE+Dice and boundary losses remain. It adds no
prediction-side FP term, margin search, sampler change, new head, or inference
routing. The comparator is the locked compute-matched Model406 v3 control.

Protocol: fixed 4-fold specimen-group split, 70 train images / 59 groups;
matched initialization/recipe/inference; native-resolution image-wise
four-class mIoU with empty-union=0; 20,000 paired specimen-group bootstrap
draws. Read the registered gate and inherited Model392 safeguards from the
protocol; do not substitute a different success criterion. A pass permits only
the separately locked seed-2 grouped-OOF confirmation, not an official-valid
or test evaluation for tuning.

Verified completed train-OOF evidence (do not treat as independent validation):
- Candidate 79.0401128%, matched Model406 v3 control 79.0112214%, paired
  delta +0.0288914 pp, group-bootstrap 95% CI [+0.0087219,+0.0490260],
  positive folds 3/4; fold deltas [+0.04237,+0.06236,-0.00346,+0.01154] pp.
- Candidate-vs-control class-IoU deltas in Al/Al3Ni/Primary-Si/Eutectic-Si
  order: [+0.01878,+0.02651,-0.00237,+0.07265] pp; Step4 +0.05946 pp.
  Candidate-vs-Model392 parent +0.06262 pp (95% CI [-0.13320,+0.27543]);
  this parent comparison is not the primary comparison.
- Registered gate is NO-GO: required gain >=+0.20 pp failed; unmatched tiny
  predicted-component non-increase failed for both target classes. Other
  registered checks passed, including positive CI lower bound and >=3/4
  positive folds.
- Native 8-connected GT component hit-rate (>=50% pixels predicted correctly),
  tiny [1,64) px: Al3Ni +1.370 pp [1.106,1.611], Eutectic Si +2.248 pp
  [1.979,2.519]; medium [64,256) px: +0.420 and +0.233 pp respectively.
  Descriptive mean component recall deltas in native bins [1,3]/[4,15]/[16,63]
  px were Al3Ni [+2.161,+1.928,+0.735] pp and Eutectic Si
  [+5.780,+3.586,+1.241] pp.
- Unmatched predicted components of native size 1-63 px per image rose from
  317.87 to 341.59 for Al3Ni (delta +23.71, 95% CI [+19.42,+28.22]) and
  207.44 to 234.91 for Eutectic Si (delta +27.47, CI [+21.51,+33.69]);
  target FP pixels/image rose +172.5 and +449.3 respectively.
- A separate descriptive FP-area audit reused the generic Model404 analyzer
  with Model408 candidate / Model406 control / Model392 parent paths (the JSON
  retains the generic Model404 experiment label; it is not an adoption gate).
  Total predicted 1-63px components/image rose +34.94 Al3Ni and +41.61
  Eutectic Si, of which +23.71/+27.47 were zero-overlap components. FP pixels
  inside the tiny-bin predicted components rose +200.41/+288.87; zero-overlap
  component area rose +97.90/+200.77. The remaining partially overlapping
  tiny-bin FP pixel point estimates also rose +102.51/+88.10.
- Step-stratified descriptive deltas (Step4 / non-Step4) have the same positive
  FP direction: Al3Ni false-positive pixels +276.7/+107.0 per image and tiny
  unmatched components +28.15/+20.93; Eutectic-Si false-positive pixels
  +432.6/+459.7 and tiny unmatched components +25.07/+28.98. Treat as
  exploratory subgroup summaries, not evidence for a learned route; errors
  increased in both strata.
- Deterministic train-crop exposure audit, 32 crops/fold/epoch x 8 epochs,
  found augmented-size [4,16) pixel crops containing >=1 target component in
  63.3-76.2% of probes for Al3Ni but only 30.1-36.7% for Eutectic Si. This is
  descriptive exposure evidence only, not an independent performance test and
  not permission to tune on this OOF set.

Optional primary-research context to assess critically, not to copy blindly:
- Mun et al., WACV 2024, [Small Objects Matters in Weakly-Supervised Semantic
  Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html)
  use size-balanced cross-entropy with an EWC-style staged strategy to retain
  large-object performance. Their setting is weakly supervised natural-image
  instance segmentation and includes an instance-aware metric; KoMaP uses dense
  metallography masks and image-wise four-class mIoU. Model408 already balances
  a component auxiliary over size bins, so explain whether the paper offers a
  genuinely different mechanism rather than just more small-instance weight.
- Du et al., CVPR 2022, [Pixel-to-Prototype
  Contrast](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Weakly_Supervised_Semantic_Segmentation_by_Pixel-to-Prototype_Contrast_CVPR_2022_paper.html)
  describe semi-hard prototype mining and hard-pixel sampling. This is weakly
  supervised WSSS; global supervised contrastive training has already failed
  a KoMaP OOF gate (Model403). Consider only a local, fixed-confuser mechanism
  if it is clearly distinct and can address Model408's recall/false-island
  tradeoff.
- A 2026 aluminum-alloy LOM study,
  [Implementation of Deep Learning-Based Segmentation for Quantitative
  Microstructure Analysis](https://link.springer.com/article/10.1007/s40192-026-00462-5),
  discusses eutectic-Si/intermetallic ambiguity in optical metallography and
  warns that minimum-size filtering may remove valid fragments. Its alloys,
  annotations, and object-detection metrics differ from KoMaP; use only as a
  caution against assuming every tiny island is noise.

Before interpreting results, verify source/calibration/OOF locks, one-shot
claim, full 70-image/59-group coverage, input hashes, and matched-control
provenance. If any integrity check fails, stop and report it. Do not use
per-fold intermediate metrics to select a treatment.

Read the verified summaries and these files:
- experiments/model408_tiny_confusion_preregistration.md
- experiments/model408_tiny_confusion_loss.py
- experiments/model408_tiny_confusion_train.py
- experiments/model408_tiny_confusion_calibrate.py
- experiments/model408_tiny_confusion_lock.py
- experiments/score_model408_tiny_confusion_oof.py
- experiments/model408_tiny_confusion_audit.py
- experiments/model408_tiny_component_exposure_audit.py and its result, if
  completed; treat it as descriptive crop exposure only
- experiments/results/model408_tiny_confusion_source_lock.json
- experiments/results/model408_tiny_confusion_calibration_lock.json
- the completed Model408 OOF summary, lock, and tiny-component audit
- experiments/results/model408_tiny_component_exposure_audit.json
- experiments/results/model408_predicted_component_fp_audit.json (descriptive
  output from the reused generic analyzer; interpret its input paths as above)
- the locked Model406 v3 control result/lock and Model392 parent summary used
  by the scorer
- model_results.md sections for Models 364, 385, 392, 397, 402-408
- experiments/results/model397_fixed_b4_model392_blend_oof.json and its lock
- experiments/results/model405_small_component_fixed_blend_oof.json and its
  preregistration
- experiments/model407_tiny_component_audit.py and its locked confusion audit

Return concise, evidence-grounded answers:
1. Is there a critical leakage, split, scoring, implementation, or lock defect?
2. Did Model408 pass every registered and inherited gate? Give exact paired
   mIoU delta/CI/fold consistency, class and Step4 changes, tiny/small GT hit
   rates, and unmatched 1-63px predicted-component safeguards.
3. Which small-component failure mode remains supported by OOF evidence?
   Separate component counts, pixel mass, recall, false positives, and mIoU;
   do not infer mIoU gain from component recall alone.
4. Recommend at most one genuinely distinct next experiment, or no-go, based
   on the residual errors and negative prior experiments. If recommending a
   change, state the matched grouped-OOF control, primary metric, safeguards,
   and stop/adoption rule. Do not forecast an mIoU gain.
```
