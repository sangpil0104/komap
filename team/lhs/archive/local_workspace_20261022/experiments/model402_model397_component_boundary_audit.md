# Model402: Model397 grouped-OOF component and boundary error audit

Status: exploratory, train-only CPU diagnostic. It trains no model, reads no
official valid/test data, and does not authorize a sampler, decoder, or
boundary-refinement training run.

## Question

Model388 found oracle headroom in 64–<256 native-pixel Al3Ni/Eutectic-Si
components on Model382 OOF, while Model395's component-balanced sampler
failed its realized crop-exposure gate. Those results do not show that the
same components remain difficult under the newer fixed Model397 blend.
Model402 measures, on the already locked Model397 B4/HRNet grouped train-OOF
predictions, where false negatives occur by GT component area and distance to
the GT class boundary, and the image-wise mIoU oracle headroom if each such
cell were corrected perfectly.

This is a post-selection diagnostic on artifacts already used to screen
Model397. Oracle headroom is not an attainable-gain estimate. Bootstrap
intervals describe variation over the fixed held-out groups conditional on
these predictions; they are not independent confirmation.

## Fixed inputs and metric

- Use only the 70 `data_model322_group_cv/oof_fold{1..4}/masks` images and
  masks, the exactly corresponding held-out Model329 B4 OOF probabilities,
  and both Model392 seed OOF probability archives already bound by the
  Model397 lock.
- Reconstruct the three fixed arms exactly: B4, `H=0.5*seed1+0.5*seed2`,
  and `C=0.5*B4+0.25*seed1+0.25*seed2`. Do not tune weights or thresholds.
- Report official image-wise four-class mIoU with empty-union IoU fixed to 0.
  For error localization, use 8-connected GT components and area bins
  `[1,64)`, `[64,256)`, `[256,1024)`, `[1024,4096)`, `[4096,+inf)` native
  pixels; use Euclidean GT-class interior distance bins `[0,2)`, `[2,4)`,
  `[4,8)`, and `[8,+inf)` native pixels.
- For each arm/class/bin, report GT support, false-negative pixel rate,
  component count/miss rate where applicable, and the per-image mIoU change
  under the oracle that corrects only GT pixels in that cell. Include
  fold-specific summaries and a fold-stratified specimen-group bootstrap for
  the oracle image-mean delta. Do not emit per-image scores.
- Verify Model397's immutable source lock, result, exact fold membership,
  Model392 manifests, B4 recipe args, archive IDs and file hashes. The new
  analysis has its own exclusive input lock and claim; refuse overwrite or
  rerun after the claim is written.

## Interpretation and next decision

The audit can test whether the previously observed Al3Ni/Primary-Si boundary
error pattern and Eutectic-Si component-size headroom persist in Model397's
OOF blend. A high oracle delta alone does not justify an intervention.
Component-sampler training remains NO-GO unless its actual exposure changes
and held-out component miss rates support the mechanism; Model395's failed
crop-pixel dose gate is not waived. A directional boundary head remains a
separate hypothesis: it cannot address Eutectic-Si deep-interior errors, and
requires Claude review plus a separately preregistered paired grouped-OOF
training experiment before any GPU run. No valid/test score or model-selection
decision follows from Model402.

## Completed result (2026-10-05)

The fixed Model397 blend C reproduced `79.9858620%` image-wise mIoU on its 70
grouped train-OOF images (59 specimen groups); this is not an official-valid
score. Class image-means were Al `93.0701%`, Al3Ni `80.9244%`, Primary Si
`86.9460%`, and Eutectic Si `59.0029%`.

Eutectic-Si class image-means across the fixed OOF arms were B4 `56.988%`,
HRNet seed-mean `58.046%`, and blend C `59.003%`. Although C is `+2.015pp`
above B4, small-component misses remain around 89–91%, and 64–256px misses
remain around 99%; the class-mean gain does not eliminate fragmented-region
errors and is not evidence of valid-set generalization.

Eutectic-Si 8-connected components showed miss rates of `89.83%` for 1–64px
components and `99.19%` for 64–256px components. Their perfect-correction
oracle deltas were `+2.622pp` (group-bootstrap 95% CI `[+2.067,+3.228]`) and
`+2.138pp` (`[+1.885,+2.395]`), respectively. Al3Ni and Primary-Si FN rates
within 2px of their GT boundaries were `27.35%` and `27.38%`, with oracle
deltas `+4.445pp` (`[+3.899,+5.026]`) and `+1.313pp` (`[+1.044,+1.623]`).
Eutectic-Si FN rate was also high beyond 8px from its boundary (`91.55%`), but
the corresponding oracle delta was only `+0.192pp` (`[+0.074,+0.383]`) due to
limited support. This does not mean the errors are solved or that these upper
bounds are attainable. Cellwise deltas are not additive.

The result supports separating boundary-focused Al3Ni/Primary-Si hypotheses
from Eutectic-Si component/texture hypotheses; it does not demonstrate either
will improve mIoU. No sampler gate is waived, no training began, and valid/test
were not read. Result SHA-256:
`e42822133d6a60b4a19164aed97e8bf8f7bb3434f726635b288ad24d0d3a0bcf`.
