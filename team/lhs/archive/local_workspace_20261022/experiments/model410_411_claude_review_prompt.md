# Read-only Claude Code review: post-Model410/411 direction

Use this prompt only after the Claude Code quota resets. Do not run now if
the CLI reports a usage limit.

```text
Act as an independent segmentation researcher. Give a read-only methods
review; do not edit files, run commands, access data/checkpoints/OOF maps,
inspect logs, train/evaluate models, or ask about GPU allocation.

Goal: improve KoMaP's official validation image-wise mean mIoU toward 80%.
The metric is four class IoUs per image, empty union = 0, averaged within
image and then equally across images. The repeatedly reused 20-image valid
set is not an independent estimate; do not recommend tuning against it.
The current experiment record lists a historical Model364 official-valid
observation of 79.9430% on that reused set; treat it as selection-conditional,
not independent confirmation. The development target remains 80%.

Problem focus: tiny Al3Ni and Eutectic-Si islands are often confused with Al
or with each other. Recent train-only grouped-OOF evidence:
- Model404 component-recall loss: about +0.032 percentage points mIoU,
  below its +0.20 pp gate; small GT hit rates improved but tiny unmatched
  predicted components increased. NO-GO.
- Model406 component recall + pixel hard-negative: about +0.020 pp; unmatched
  tiny predicted components increased by roughly 18/image for each target.
  NO-GO.
- Model407 predicted-component FP penalty: about -0.087 pp; it reduced tiny
  false positives but also reduced small GT hit rates by about 2–4 pp. NO-GO.
- Model408 target-confusion hinge: about +0.029 pp, still below gate, with
  additional unmatched tiny predictions. NO-GO.
- Earlier global SupCon, generic boundary loss, hard-crop mining, and generic
  pairwise-margin experiments have already been tried; do not repackage them.
- A broader resolution intervention has also been tested: Model392's
  width-normalized HRNet-W48/U-Net++ two-seed, four-fold train-OOF delta was
  +0.1042 pp (group-bootstrap 95% CI [-0.4175,+0.6007] pp), with 2/4 pooled
  folds positive and its preregistered gate failed. Do not recommend another
  generic upscaling/scale-normalization run as the next experiment.
- Separate Model392 train-OOF diagnostics found a strong boundary
  localization signal (group-macro AP 92.0%, 95% CI [91.18,92.80]); a
  train-mask-only boundary-to-interior offset-support audit found 71.75–75.31%
  support for Al3Ni and 94.11–95.49% for Primary Si at clearance >=4px and
  offsets <=16px. This is target availability/localization evidence, not
  semantic mIoU benefit; Eutectic-Si support was not assessed.

Model410's locked 70-image/59-group four-fold train-OOF screen is complete and
NO-GO. Candidate image-wise mIoU was 79.0771%, matched control 79.0114%, delta
+0.0657 pp; paired group-bootstrap 95% CI [-0.0633,+0.2158] pp; only 2/4 folds
were positive and 42/70 images improved. The largest group's positive
contribution share was 87.1%. Class deltas (Al/Al3Ni/Primary Si/Eutectic Si)
were +0.0502/+0.1250/+0.0220/+0.0657 pp; Step4 delta was -0.0406 pp.
Width-normalized component diagnostics: Al3Ni GT-hit-rate delta was
-0.536/-0.554/-0.655 pp over bins [1,4), [4,16), [16,64), while unmatched
predicted islands changed -1.40/-0.80/-2.26 per image. Eutectic-Si hit-rate
delta was +0.252/+0.585/+0.464 pp, while unmatched islands changed
-0.014/+0.443/+1.657 per image. The fixed gate failed on effect size,
uncertainty, fold consistency, Al3Ni hit-rate safety, and non-increasing
unmatched predictions. Official valid/test were not read. Treat this only as
development evidence, not a generalization claim.

Model411's locked 70-image/59-group four-fold train-OOF run is complete and
NO-GO. It compared a single HRNet-W48/U-Net++ component-balanced bidirectional
loss fine-tune against a matched continuation control, initialized from
Model392 seed 1. Candidate-minus-control image-wise mIoU was +0.0156 pp
(paired group-bootstrap 95% CI [+0.0066,+0.0249] pp), positive in 4/4 folds
and 46/70 images; the largest group's positive-contribution share was 19.8%.
Class deltas (Al/Al3Ni/Primary Si/Eutectic Si) were
+0.0155/+0.0285/+0.0086/+0.0096 pp; Step4 was +0.0243 pp. It failed the
pre-registered +0.20 pp effect-size gate and safety gates because target
false-positive pixel burden and unmatched tiny predicted components increased
for both Al3Ni and Eutectic Si. Unmatched tiny-component deltas were
+23.24/+16.59 per image, and target FP-pixel deltas were +201.16/+30.47 per
image. Tiny GT hit-rate deltas were +1.33 pp for Al3Ni and +1.00 pp for
Eutectic Si; the 64–255 px bins changed +0.42 pp and -0.14 pp respectively.
Against its Model392 initialization parent, delta was +0.0493 pp with group
bootstrap 95% CI [-0.1509,+0.2624] pp, 3/4 positive folds, and a 90.8%
largest-group positive-contribution share; Step4 delta was -0.1615 pp. This
parent comparison is uncertain, not independent evidence of improvement.
Official valid/test were not read; locked OOF artifacts and the hash chain
were verified. BiCC is only a hypothesis source (3D medical lesions,
different metric), not KoMaP evidence.

Based only on the facts above, answer:
1. Does Model410 falsify component-balanced crop exposure as a useful lever,
   or only this exact sampler under this recipe? Explain what the grouped
   results support and what they do not.
2. Taken together, do Model410 and Model411 falsify component-count based
   sampling/losses, or only these specific implementations? Explain the
   evidence and uncertainty without recommending another near-duplicate.
3. Reassess your first suggestion of changing input scale in light of the
   Model392 result above; do not recommend repeating it. Then evaluate one
   genuinely different single-factor supervised experiment—possibly a
   boundary-to-interior direction auxiliary task, but recommend it only if
   target support makes it plausible. Specify the paired control, predicted
   class/component error signature, grouped train-OOF metric, effect-size and
   safety/adoption gates, and falsification condition. If a train-only
   feasibility audit is needed first, define it and its stop condition.
4. Identify measurement pitfalls that could make component hit/FP changes
   appear useful while image-wise mIoU does not improve.

Keep the response concise (at most 700 words). Do not infer, request, or use
any Model409 score or per-fold result. Do not access official valid/test data
or use them for threshold, class-weight, or model selection.
```
