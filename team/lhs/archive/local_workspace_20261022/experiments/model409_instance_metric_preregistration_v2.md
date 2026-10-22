# Model409 revision 2: tiny-component local feature separation

## Revision history and independent review

Revision 1 completed CPU/P100 syntax and loss smoke checks, fixed-fold
validation, and a 32-crop/fold outer-train-only gradient calibration attempt.
It did not perform any optimizer step or produce a checkpoint. The calibration
reached folds 1–3 but repeatedly ran out of memory in fold 4. Review of the
autograd graph found that revision 1 called `grid_sample` on a full projected
feature map once per connected component, retaining repeated maps for backward.
Revision 2 changes only feature-coordinate sampling: all positive/negative
coordinates are concatenated and sampled in one operation per image. The
objective, coordinates, sampling caps, projection architecture, optimizer,
folds, and decision gates are unchanged. A dense 392-component stress test is
required before GPU calibration. Revision-1 source lock and logs are retained;
revision 2 receives a separate immutable source/calibration/OOF lock chain.

Prior experiments constrain the question: Model403 attracted arbitrary
same-class pixels across instances; Model404/406 raised GT-component recall
but increased unmatched tiny predictions; Model407's component-FP penalty
also suppressed true tiny recall; Model408's direct confuser-logit hinge
improved small-component recall but failed the score/island gates. Model409
still tests a different hypothesis: change decoder feature geometry using
same-instance positives and nearby, class-specific confuser negatives, rather
than directly changing target logits or inference-time predictions. This is
exploratory and has no expected-score claim.

## Frozen treatment

- Parent: Model392 seed-1 HRNet-W48/U-Net++ checkpoint in each fixed
  specimen-group fold. Fine-tune for exactly 8 epochs with patch 512,
  width-normalized input scale 2, per-item paired crop/augmentation stream,
  CE+Dice, boundary loss, AdamW, and cosine schedule as Model404.
- For each post-augmentation CPU mask, find 8-connected Al3Ni/Eutectic-Si
  components of area `[4,256)` augmented pixels; exclude crop-border regions.
  Same-component pixels are the only positives. Confuser pixels are only Al
  for Al3Ni, and Al/Al3Ni for Eutectic Si, inside the 2–6px spatial annulus.
  Deterministic caps are 32 positive and 64 negative coordinates/component.
  Fixed contrastive temperature is 0.2; component losses are averaged within
  crop × class × size-bin cells.
- A training-only pointwise projection head (decoder channels → at most 128 →
  32) is discarded for inference. It does not change the semantic checkpoint
  interface. There is no sampling-policy change, logit loss, prediction-side
  FP loss, new inference branch, ensemble, or routing.
- Outer-train-only calibration uses 32 crops/fold, zero optimizer steps, and a
  fixed 10% median decoder-gradient ratio. Insufficient active support or
  unsafe coefficient means stop before training.
- Compare once to the fixed Model406 v3 matched continuation control and retain
  Model392 seed-1 as parent safeguard. Evaluate the exactly-once 70-image,
  59-specimen-group, four-fold train OOF using image-wise mean of four class
  IoUs and `empty=0`; use 20,000 paired specimen-group bootstrap draws.

## Adoption gate

All gates are required: candidate-control gain ≥`+0.20` percentage points,
paired group-bootstrap CI lower bound >0, at least 3/4 folds positive, worst
fold ≥`−0.30`pp, no target-class regression, Step4 ≥`−0.30`pp, candidate not
below the Model392 parent, tiny/small GT component hit-rates nondecreasing,
and unmatched 1–63 native-pixel predicted components nonincreasing for both
target classes. Even a pass only permits independently locked seed-2 OOF
confirmation. Official valid/test, threshold tuning, coefficient search,
ensemble, and routing are excluded.

## Data/resource scope

Only the fixed train-group splits are used for calibration/training/OOF; OOF
`valid` refers only to each held-out subset of `train.zip`. Official
`valid.zip` and `test.zip` remain unread. Use the existing P100 `komap`
environment; do not terminate or interfere with other server jobs. Source v2
must be hash-locked and stress-tested before any optimizer step.
