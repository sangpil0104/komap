# Model406 v3: small-component recall with hard-negative island suppression

This is the paired grouped-OOF evaluation of the fixed Model406 treatment. The v2 low-memory calibration has completed; this revision does not change its candidate loss, crop sampling, four fixed grouped folds, parent checkpoints, optimizer, schedule, inference, or adoption gates.

## Treatment and control

- Candidate auxiliary loss: Model404 v4 balanced recall loss for Al3Ni and Eutectic-Si GT components, plus a top-0.1% hard-negative island penalty outside a one-pixel same-class GT halo.
- Candidate loss coefficient: the fold-specific value in the immutable v2 calibration file, derived from 32 one-crop outer-train probes and a 10% target auxiliary/main decoder-gradient ratio. No optimizer step or held-out image was used for calibration.
- Control: matched 8-epoch continuation using CE+Dice+boundary only.
- Both arms start at the same Model392 seed-1 checkpoint per fold and use identical data sampling, architecture, optimizer, schedule, and inference.

## Evaluation and gate

Use the fixed 4-fold, 70-image/59-specimen-group train OOF set; official image-wise four-class mIoU with empty-union=0; and 20,000 paired specimen-group bootstrap draws. Report class IoUs, Step4, tiny/medium GT component hit rates, and zero-overlap tiny predicted-component counts for Al3Ni and Eutectic Si.

Adopt only if every inherited Model404 v6 paired gate passes: mIoU gain at least +0.20 percentage points, paired group-bootstrap lower bound above zero, at least 3/4 positive folds, worst fold no lower than −0.30 points, all class/Step4 guards, targeted tiny/medium component-hit guards, and non-regression versus the fixed Model392 seed-1 parent. Additionally, unmatched zero-overlap 1–63px predicted components/image must not increase for either target class versus the matched control.

There is one fixed candidate and no coefficient, threshold, or blend search. This is train-only grouped OOF; official valid and test remain sealed. A pass authorizes only an independently locked seed-2 grouped-OOF confirmation.
