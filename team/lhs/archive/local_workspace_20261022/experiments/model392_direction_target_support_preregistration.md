# Model392 direction-target support audit

## Purpose

Before considering a direction-to-same-class-interior head, quantify whether
the proposed targets have stable support in each training fold. This is a
mask-only geometry audit, not a model experiment: it trains no model, reads no
OOF predictions, computes no mIoU, and cannot authorize training.

The candidate classes are fixed in advance as Al3Ni (class 1) and Primary Si
(class 2), based on the earlier Model387 observation that their false
negatives concentrate near GT boundaries. Eutectic Si is not a candidate
boundary correction because its recorded errors are predominantly deep
interior.

## Frozen geometry

- For each of the four fixed specimen-group folds, use only its outer-training
  masks (52–53 images); verify zero specimen-group overlap with held-out OOF
  groups. Do not read any official valid/test files or predictions.
- A source pixel is a class-c boundary pixel when it is labeled c and is in
  the same class-agnostic 3x3 GT boundary band used by the Model392 auxiliary
  loss.
- For each candidate class, define possible same-class interior targets as
  pixels whose exact Euclidean distance to the complement of that class is at
  least 2, 4, or 8 pixels. For each source pixel, find the nearest such target.
- Record support at maximum source-to-target distances 4, 8, 16, and 32 px.
  Quantize supported target vectors into eight fixed compass bins
  N, NE, E, SE, S, SW, W, NW. Report counts, support fractions, and specimen
  groups contributing to every class/fold/clearance/radius/direction cell.
- The primary descriptive table is clearance 4 px and maximum offset 16 px.
  The complete predeclared sensitivity grid is reported without selecting a
  setting from model performance.

## Interpretation

This audit determines whether a direction target is geometrically available
and whether direction bins are represented across groups. It does not show
that a network can learn those targets or that using them improves semantic
segmentation. Sparse or fold-concentrated support is a reason to defer or
restrict a direction head; abundant support is only a feasibility condition.
No training, valid evaluation, or test use follows automatically.

Before execution, create and verify
experiments/results/model392_direction_target_support_lock.json. It binds the
protocol and script, the Model392 sequence/runtime locks, grouped split
inventory, and hashes of every consumed outer-training mask. The report is a
one-shot artifact and refuses overwrite.
