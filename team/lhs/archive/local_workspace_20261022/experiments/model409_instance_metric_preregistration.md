# Model409: tiny-component local feature separation

## Independent review and motivation

Model403's contrastive anchors were sampled per class/crop, and every sampled
same-class pixel became a positive regardless of instance or location. Its
completed folds 1 and 2 were both below the paired Model392 reference, making
the preregistered 3/4-positive-fold gate unreachable; training was stopped.
Model404/406 increased tiny GT-component hit rate but did not meet the mIoU
gain gate and added unmatched tiny predictions. Model407's predicted-component
penalty reduced real tiny-target recall. Model408's direct confuser-logit hinge
raised tiny recall and false-positive mass together, missing its adoption gate.
These results argue against another logit margin, positive-only recall weight,
generic island penalty, crop sampler, routing, or ensemble search.

The single new hypothesis is that a training-only embedding objective can
separate the feature representation of a small Al3Ni/Eutectic-Si GT component
from the immediately surrounding phase confusers without directly raising its
class logits everywhere. This is distinct from Model403's cross-instance
same-class attraction and Model408's logit hinge: positives are restricted to
one connected GT component; negatives are fixed confuser-class pixels in the
2–6 augmented-pixel annulus around that component. The projection head is
discarded at inference. This is an exploratory hypothesis, not an expected
gain; the tiny dataset and morphology variability make false-negative and
over-separation risks substantial.

## Fixed protocol

- Model392 seed-1 HRNet-W48/U-Net++ checkpoint per fixed specimen-group fold;
  eight-epoch fine-tuning, patch 512, width-normalized input scale 2, same
  per-item crop/augmentation stream and optimizer/scheduler as Model404.
- No changes to CE+Dice, boundary loss, crop sampling, semantic architecture,
  or inference. A two-layer pointwise projection head (decoder channels → at
  most 128 → 32) is training-only. Its initialization is isolated from the
  global RNG. The exact shared runner computes it in the candidate only.
- Eligible post-augmentation CPU-mask GT components are 8-connected target
  classes Al3Ni (1) and Eutectic Si (3), area `[4,256)` pixels, crop-border
  excluded. Fixed confusers are Al for Al3Ni and Al/Al3Ni for Eutectic Si.
  Positive pixels are sampled deterministically (max 32/component); confuser
  negatives are from the 2–6px annulus (max 64/component). Temperature is 0.2.
  The supervised contrastive denominator contains same-component positive
  pixels plus these local confusers. Component losses are macro-averaged over
  crop × class × area-bin cells. No prediction-derived region is used.
- Calibrate one coefficient per outer fold using 32 outer-train crops, with
  zero optimizer steps, to make the auxiliary decoder-gradient norm 10% of
  CE+Dice+boundary. No held-out groups, official valid, or test data in
  calibration.
- Compare the one-shot candidate against the fixed Model406 v3 matched
  continuation control, and retain Model392 seed-1 as a parent safeguard.
  Evaluate exactly-once four-fold OOF on 70 train images / 59 specimen groups
  using image-wise mean of four class IoUs, empty-union=0, plus 20,000 paired
  specimen-group bootstrap draws.

## Frozen decision gate

Adopt only if candidate-control gain is at least `+0.20` percentage points,
paired group-bootstrap 95% CI lower bound is positive, at least 3/4 folds are
positive, worst fold is no worse than `−0.30`pp, no target class regresses,
Step4 is no worse than `−0.30`pp, candidate is not below its Model392 parent,
tiny/small GT component hit rates do not decrease, and unmatched 1–63 native
pixel predicted components do not increase for either target class. A pass
permits only an independent-seed OOF confirmation. No official valid/test
evaluation, inference routing, model blending, coefficient search, or
post-hoc threshold tuning is authorized by this experiment.

## Current status

Independent source review and P100 availability check completed on
2026-10-06 KST. No training was active before this experiment. Source and
calibration locks must be created and verified before the first optimizer
step. Any insufficient local-confuser support, nonfinite loss/gradient,
resource failure, or incomplete fold invalidates adoption; preserve artifacts
and do not replace them with a more favorable rerun.
