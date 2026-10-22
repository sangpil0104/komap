# Model405: small-component branch swap on fixed grouped OOF

## Question and scope

Model404 v6 improves tiny Al3Ni and Eutectic-Si GT-component hit rate but creates many small false-positive islands, leaving little net image-mIoU gain. Model405 tests exactly one pre-fixed probability blend to see whether the existing Model397 branches suppress those islands:

- Reference: 0.50 × Model329 B4 + 0.25 × Model392 seed1 + 0.25 × Model392 seed2 (the fixed Model397 blend).
- Candidate: 0.50 × Model329 B4 + 0.25 × Model404 v6 component-recall candidate + 0.25 × Model392 seed2.

This is a single branch substitution. There is no coefficient, class-bias, threshold, or routing search. Evaluation is CPU-only on the exact 70-image/59-specimen-group, four-fold train OOF set. Official validation and test inputs are excluded. Because this follows inspection of development OOF results, it is selection-conditional evidence, not an independent generalization estimate.

## Frozen measurements

- Primary metric: image-wise mean of four class IoUs, empty union = 0, using the official class order Al, Al3Ni, Primary Si, Eutectic Si.
- Paired uncertainty: 20,000 specimen-group bootstrap draws within each of the four fixed folds, seed 405.
- Targeted component hit: for each class separately (Al3Ni, Eutectic Si), an 8-connected GT component of 1–63 native pixels is hit when at least 50% of its pixels are predicted as that class.
- Tiny predicted-component diagnostic: for each target class, count 8-connected predicted components of 1–63 pixels and the subset with zero overlap with same-class GT.
- Also report target-class FP/FN pixels, per-class IoU deltas, fold deltas, and Step4 delta. No measurement is used to search or alter the blend.

## Adoption screen

All conditions must hold to call the branch swap promising:

1. Candidate image-mIoU gain over the fixed Model397 reference is at least +0.20 percentage points, with paired group-bootstrap 95% CI lower bound above zero.
2. Gain is positive in at least 3/4 folds and the worst fold is at least −0.30 points.
3. Each target-class IoU delta is nonnegative; no class delta is below −0.50 points; Step4 is at least −0.30 points.
4. Tiny GT-component hit rate does not decrease for either target class.
5. Mean zero-overlap tiny predicted components per image do not increase for either target class.

This screen only decides whether to run a separately preregistered independent-seed grouped-OOF confirmation. It never authorizes an official-valid or test evaluation. If any condition fails, stop and retain the Model397 blend.

## Integrity and execution

The lock binds this protocol and scorer, the existing Model397 locked inputs/result/claim, the Model404 v6 lock/result and prior FP audit, each fold's fixed masks, and the B4/Model392-seed2/Model404-candidate probability archives plus candidate checkpoint/args/OOF summaries. The scorer verifies the existing Model397 and Model404 source locks, exact ID agreement, finite normalized probabilities, native mask shape, and 70/59 OOF coverage before scoring. It refuses to overwrite a lock, score claim, or result.
