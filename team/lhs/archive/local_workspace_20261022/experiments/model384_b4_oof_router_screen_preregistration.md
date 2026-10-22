# Model384: B4 train-OOF shrinkage-router screen

## Question and scope

Does a deliberately low-capacity image-level router between Model322 (team B4)
and Model329 (fullstats/width-4000 B4) show enough train-OOF signal to justify
further routing work? This is a CPU-only **screen**, not a confirmatory estimate
and not a candidate submission. Official `valid.zip` and `test.zip` are not read.

## Fixed inputs and metric

- Exactly-once train OOF probability maps from
  `model322_team_b4_group_oof/train_oof_probs.npz` and
  `model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz`.
- Matching masks only from `data_model322_group_cv/oof_fold1..4`.
- Require exact equality of both OOF key sets with the 70 masks, 59 specimen
  groups, and group-disjoint base folds.
- Official image-wise mean of four class IoUs; empty union is zero.
- Fixed global baseline is the equal probability mean, `0.5 * Model322 +
  0.5 * Model329`.

## Router search, fixed before scoring

- One image-level threshold only; route endpoint is Model322 or Model329.
- Features from the global blend only: predicted-present class count (area at
  least 0.1%), predicted Eutectic-Si area fraction, and normalized mean entropy.
- Threshold quantiles `{0.25, 0.50, 0.75}`, both directions, and shrinkage
  lambda `{0.25, 0.50, 1.00}`. Probability output is the global blend moved by
  lambda toward the selected expert. Lambda zero is the fixed baseline and wins
  exact ties.
- Outer splits are the four existing base OOF folds. In each outer training
  portion, choose one rule by specimen-group LOGO inner scoring; fit its feature
  threshold on that outer training portion and score only its held-out base fold.
- Report paired image deltas, 20,000 specimen-group bootstrap draws, and the
  four outer-fold deltas. No post-result grid changes.

## Leakage limitation and decision gate

The inner-router training images' OOF predictions may come from base fold models
that trained on an outer-heldout group. Therefore outer results are not an
independent generalization estimate and are not proof of routing benefit. A
positive result only prioritizes a later confirmation using an eligible full
OOF family; a negative result deprioritizes this small router. Screen pass
requires all three: pooled image-weighted delta at least `+0.30 pp`, group
bootstrap 95% CI lower bound above zero, and positive delta on at least 3/4 outer
folds. Do not inspect valid/test labels or choose any parameter from them.

## Execution

`experiments/model384_b4_oof_router_screen.py`; CPU-only via
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
Output is a new, fail-if-exists JSON under
`outputs/model384_b4_oof_router_screen/results.json`.
