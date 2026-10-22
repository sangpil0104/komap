# Claude Code 독립 검토 요청 — Model404 v6 작은 성분 교정

아래 검토는 Claude Code 주간 한도가 초기화된 뒤 한 번만 제출한다(안내된 시각: 2026-10-06 23:00 KST). 초기화 전 우회·재시도하지 않는다. 읽기 전용 검토만 요청한다. 파일 수정, 학습 실행, 공식 validation/test 접근, GPU 점유·메모리 질문은 금지한다.

```text
You are an independent semantic-segmentation researcher. Review the exact
locked Model404 v6 Al3Ni + Eutectic-Si tiny-component recall experiment in
this repository. Do not edit files or run training. Do not access official
validation/test or propose threshold/routing sweeps. Do not ask about GPU
occupancy.

Official metric: for each image average IoU over four classes, then average
images; empty-union class IoU is zero. The current raw official-valid leader
is 79.94297% on a repeatedly reused 20-image set, so it is selection
conditional and must not be tuned further. Test remains sealed.

Fixed Model397 train-only grouped OOF evidence (70 images / 59 specimen
groups; development diagnostics, not independent confirmation):
- Native-grid 8-connected Al3Ni components of 1–<64 px: 66,342 components.
  In the fixed Model397 blend, 40.83% have <50% true-class recall and 25.18%
  are completely missed. Over component pixels, 58.57% predict Al3Ni,
  26.26% Al, 12.86% Eutectic Si, and 2.30% Primary Si.
- Tiny Eutectic-Si components of 1–<64 px: 43,545 components. 31.90% have
  <50% recall and 21.51% are completely missed. Their pixels predict
  Eutectic Si 66.06%, Al3Ni 21.91%, Al 11.46%, and Primary Si 0.57%.
- An ideal per-class oracle that fixes every selected-region error has
  image-mIoU upper bounds of +1.7374 pp for tiny Al3Ni and +2.6216 pp for
  tiny Eutectic Si. These separate idealized upper bounds are not expected
  gains and must not be added.

The locked v5 treatment adds component-normalized -log(mean p(true class))
for augmented 512-grid, 8-connected GT components of classes 1 (Al3Ni) and
3 (Eutectic Si), using area bins [4,16), [16,64), [64,256) px. At nominal
2x linear input scale these approximately correspond to native 1–<64 px;
width normalization and geometric augmentation make this approximate.
Crop-border-touching components are excluded. Each nonempty crop × class ×
size-bin cell is averaged equally. Existing pixel CE+Dice and boundary losses
remain to constrain precision and the other phases. Native 64–256 px
components are guard-only, as are class IoUs for Al and Primary Si.

The shared ComboLoss deliberately has no inverse-frequency CE class weights:
previous weighted experiments expanded Eutectic Si by roughly one pixel
across phase boundaries. Assess whether the spatially localized component
objective plus the retained CE+Dice/boundary terms controls that overgrowth,
or whether the loss can still reward false-positive islands.

Compute-matched control and candidate start from each fixed Model392 seed1
fold checkpoint and continue eight epochs with the same per-item-seeded
sampling/augmentation order, optimizer, schedule, and inference. A fold-local
coefficient is set from eight outer-train batches using 10% of the median
auxiliary/main shared-decoder gradient norm; calibration has zero optimizer
steps. Evaluation is paired four-fold specimen-group OOF. Adoption requires
+0.20 pp overall, paired group-bootstrap 95% CI lower >0, at least 3/4
positive folds and worst fold >= -0.30 pp; target-class IoU deltas >=0,
other-class deltas >= -0.50 pp; Step4 >= -0.30 pp; all tiny/medium
component-hit deltas nonnegative with CI lower >= -5 pp. Seed1 is not final
adoption; same gate must hold on seed2. No gain is assumed.

An independent scorer audit found that v5 could pass when the component
candidate beat its compute-matched continuation control while still being
worse than its Model392 seed1 initialization parent. Before training, v6 adds
a paired OOF point-estimate requirement: candidate-vs-parent image-mIoU delta
must be >=0.0 pp. The parent confidence interval is reported but not separately
thresholded. Assess whether this closes the adoption gap and preserves a
meaningful interpretation of the control comparison.

The v3 Eutectic-only, v4 two-class, and v5 scorer-corrected queues were stopped
before calibration or optimizer steps. v4's synthetic scorer test found and
v5 fixed a target-class label-formatting bug. The v6 source lock and CPU/P100
smoke checks passed. Model403 was subsequently stopped after full OOF folds 1
and 2 both had negative candidate-vs-parent deltas, making its preregistered
3/4-positive-fold adoption gate impossible to pass; fold3 stopped at epoch
22/70, with its checkpoints retained, and fold4 was not run. This is not a
full Model403 OOF score.

Model404 v6 completed four-fold grouped train-OOF. Its four fold-local gradient
calibrations each used 8 outer-train batches and zero optimizer steps; locked
component weights were 0.0024277721, 0.0025784487, 0.0029262254, and
0.0027533217. The 34-source v6 lock SHA-256 is
`2024be7d3a370fd0c3448e578f7276514001c28202dbc4122d99e93fd3091692`.

Official metric is image-wise mean of four class IoUs, empty-union=0; 70 images,
59 specimen groups, 4 fixed folds. Results:
- Candidate 79.0433246%, compute-matched control 79.0114458%, fixed Model392
  seed1 parent 78.9774938%.
- Candidate-control delta +0.0318788 pp, paired group-bootstrap 95% CI
  [+0.0182046,+0.0458351], positive in 4/4 folds; gate NO-GO solely because
  the preregistered minimum gain is +0.20 pp. Every other treatment/control
  gate passed. Candidate-parent point delta +0.0658308 pp, but its CI
  [-0.1321287,+0.2780864] includes zero.
- Small native 1–<64 px GT component hit-rate rose +1.8224 pp for Al3Ni
  (95% CI [+1.5914,+2.0328]) and +1.7155 pp for Eutectic Si
  ([+1.4826,+1.9541]).
- Post-hoc, read-only OOF argmax/8-connected audit found tiny predicted
  components increased by 47.1 Al3Ni and 41.9 Eutectic-Si components/image;
  zero-overlap same-class tiny components increased by 30.9 and 24.6/image.
  Target-class FP pixels rose +266/+171 per image while FN pixels fell
  -273/-142. This suggests the recall auxiliary also creates many small
  false-positive islands, explaining the small mIoU net gain.

Authoritative local artifacts:
- experiments/results/model404_component_recall_oof_v6.json
- experiments/results/model404_predicted_component_fp_audit_v6.json

Both results are train-only; official validation/test were not read. Do not
adopt Model404 v6 or use its OOF maps as independent confirmation.
Do not simply revive component-balanced crop sampling; prior Model389 E2 and
Model395 realized-dose gates are NO-GO.

At most one follow-on may be recommended after the independent review. A
specific low-cost hypothesis to assess (not yet scored) is to replace only the
25% Model392-seed1 branch in the already fixed Model397 OOF blend:
`0.50*B4 + 0.25*Model404-v6-candidate + 0.25*Model392-seed2`, compared with
the existing fixed `0.50*B4 + 0.25*Model392-seed1 + 0.25*Model392-seed2`.
This is a post-hoc, selection-conditional train-OOF screen motivated by the
small-component recall/FP trade-off, not an independent estimate. It must use
the exact same held-out group IDs, no weight/route search, and no official
valid/test. Any promising result needs independent seed2 grouped OOF before a
valid look. Assess whether this branch substitution is scientifically worth a
single fixed screen or whether its likely small/selected gain makes it a
NO-GO; do not forecast an mIoU gain. Alternatively, if rejecting it, suggest
at most one distinct mechanism (e.g. class-wise local CC loss that explicitly
scores FP context), with precise grouped-OOF gates. BiCC is binary 3D medical
lesion work, not a drop-in for KoMaP 2D 4-class. The 2026 multi-class
one-vs-rest CC/Blob study is also 3D BraTS-METS, with a Dice/PQ metric and a
reported detection-vs-overlap trade-off; treat transfer as a hypothesis.

Metallography-specific literature/history to distinguish from this target:
the GRAF paper's gap-region weighting addresses merging of adjacent primary-Si
instances on Al-Si micrographs, not class confusion of tiny Al3Ni/Eutectic-Si
semantic components. This was already screened as Model324 against Model302
train OOF at the actual 2x training resolution: the perfect-gap-correction
image-mIoU oracle headroom was only +0.0113 pp, so training was cancelled.
Do not recommend rerunning that same Primary-Si gap weighting absent new
measured evidence. [GRAF DOI: 10.1016/j.commatsci.2023.112620]. A separate
Al-Si synchrotron-CT study used pixel-wise weighting for Si and Si-aluminide
interfaces, but the tomography modality differs from KoMaP RGB metallography;
it is not evidence that more global class/boundary weighting will help here.
[Strohmann et al., 2019](https://www.nature.com/articles/s41598-019-56008-7).
Also assess whether native 1–3 px semantic islands could reflect rasterization,
annotation ambiguity, or fragmented labels rather than reliable physical
particles; 8-connectivity is an operational metric, not proof of instance
identity. Do not change the current locked run based on this concern.

Do not suggest input-scale increase alone as an untested fix. Historical
Models253–266 tested 3x/1536 patches, matched 1x, and full-image-stat 2x/3x
fine-tuning. Model253 3x scored 74.9705% versus its Model242 EMA parent
76.8640%; Model265 full-stat 3x scored 75.0537%, versus full-stat 1x Model257
77.2071%; full-stat 2x Model259 scored 76.8104%. These are old, repeatedly
reused official-validation estimates and not independent train-OOF
confirmation, so treat them as cautionary rather than decisive. Any resolution
hypothesis must add a distinct mechanism and grouped-OOF test, not merely copy
the team's reported 3x/4x recipe.

Please inspect read-only:
- experiments/model404_predicted_component_fp_audit.py (planned post-hoc OOF-only diagnostic; do not run before all locked OOF maps exist)
- experiments/model404_component_recall_preregistration_v6.md
- experiments/results/model404_preregistration_lock_v6.json
- experiments/model404_component_recall_loss_v4.py
- experiments/model404_component_recall_train_v4.py
- experiments/model404_component_recall_calibrate_v4.py
- model336_hrnet_widthnorm_candidate/komap_seg/losses.py
- experiments/score_model404_component_recall_oof_v6.py
- experiments/run_model404_component_recall_oof_v6.sh
- experiments/model404_tiny_component_confusion_audit.py
- experiments/results/model404_tiny_component_confusion_audit.json
- model_results.md (prior sampler, boundary, and component findings)

Because the locked experiment is already underway, return: (1) whether you
see a critical mechanistic, leakage, or scoring defect that justifies stopping
the exact locked run; (2) otherwise, whether the existing gates adequately
separate component recall from overprediction and parent-model regression; and
(3) at most one distinct follow-on hypothesis if the preregistered gates fail.
Do not propose changing this run, editing locked sources, or forecast an mIoU
gain. Label hypotheses as hypotheses and cite primary literature only where
needed.
```
