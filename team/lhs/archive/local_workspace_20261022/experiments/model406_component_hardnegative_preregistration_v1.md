# Model406: small-component recall with hard-negative island suppression

## Motivation

Model404 v6 slightly improved tiny Al3Ni/Eutectic-Si component hit rates but produced many additional tiny false-positive islands. The single fixed Model405 branch substitution did not improve grouped train-OOF image-mIoU and increased zero-overlap tiny predictions. Model406 therefore pairs component-recall supervision with a fixed hard-negative penalty instead of adding more positive-only pressure.

## Frozen treatment

- Targets remain Al3Ni and Eutectic Si GT components of 1–63 native pixels, identified on the post-augmentation training mask with the same 8-connected bins as Model404 v4.
- The positive term is the unchanged balanced component-recall loss.
- For each crop and target class, the negative term selects the top 0.1% target-class probabilities among non-target pixels outside a one-pixel max-pool halo around same-class GT. It applies mean negative log probability of not belonging to the target class. This is a fixed rank-based hard-negative rule, not a probability threshold or searched hyperparameter.
- The two terms are summed at unit relative scale. Their combined decoder-gradient norm is calibrated to 10% of the CE+Dice+boundary main-loss gradient using 8 outer-train-only probe batches per fold; calibration takes zero optimizer steps and uses no mIoU/held-out images.
- Control is the matched 8-epoch continuation with CE+Dice+boundary only. Both arms start from the same Model392 seed1 fold checkpoint and use identical per-item sampling, architecture, optimizer, schedule and OOF inference.

## Evaluation and adoption gate

Use the fixed 4-fold, 70-image/59-specimen-group train OOF set; official image-wise four-class mIoU with empty-union=0; 20,000 paired specimen-group bootstrap draws. Report class IoUs, Step4, tiny/medium GT component hit rates, and zero-overlap tiny predicted-component counts for Al3Ni and Eutectic Si.

Adopt only if every inherited Model404 v6 paired gate passes: mIoU gain at least +0.20 percentage points, paired group-bootstrap lower bound above zero, at least 3/4 positive folds, worst fold no lower than −0.30 points, all class/Step4 guards, targeted tiny and medium component-hit guards, and non-regression versus the fixed Model392 seed1 parent. Additionally, the candidate must not increase mean zero-overlap 1–63px predicted components per image for either target class versus the matched control.

This is one fixed candidate with no coefficient, threshold, or blend search. The OOF result is selection-conditional development evidence only. A passing screen can authorize only an independently locked seed2 grouped-OOF confirmation, never an official-valid or test look. Test stays sealed.

## Reproducibility

The source lock binds code, the Model397/Model404/Model405 provenance, raw outer-fold train/held-out symlink targets, masks, fixed group-fold configuration and Model392 seed1 checkpoints/OOF outputs. Separate locks bind the outer-train calibration and trained candidate/control checkpoints plus OOF maps before one-shot scoring. The runner refuses active experiment locks, symlinked output roots, overwrite of checkpoints, and repeat scoring.
