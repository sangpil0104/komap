# Model392 runtime-safety addendum (revision 3, pre-score)

This addendum supplements the [revision-2 amendment](model392_widthnorm_amendment.md)
and preserves its model recipe, estimand, bootstrap, and numeric gates. It records
execution safeguards found in an independent read-only review before Model392
training or any Model392 score access. The revision-2 lock remains unchanged;
the [revision-3 runtime lock](results/model392_widthnorm_runtime_lock.json)
supersedes it for all subsequent Model392 commands.

## Confirmatory OOF analysis

- Apply the amendment's pooled-result rule literally: at least three of four
  pooled fold deltas must be positive, in addition to the per-seed fold/class/
  Step4 safeguards. Two seeds each passing three folds does not imply that the
  pooled result passes three folds.
- Report the seed-2 minus seed-1 mean-delta difference and the fold-specific
  training/held-out image-width counts. These are descriptive; they do not add
  tuning choices or change the fixed gate.
- The confirmatory analysis has one fixed output path and one persistent
  exclusive claim. Claiming occurs before loading fold scores or artifacts.
  Write the completed JSON to a new temporary path and atomically rename it to
  the final output. An interrupted or failed claimed analysis is not rerun
  under this lock; any retry requires a new protocol decision.

## Full-train resume

The sequential P100 full-train runner skips a seed only when its epoch-70 state
and clean completion log agree. If epoch 70 was checkpointed but the process
did not leave the clean completion marker, the runner invokes the trainer in
resume mode to finalize that run before advancing. A cleanly completed seed is
not appended as a new training attempt when the runner is restarted, so an
interruption during seed 2 can resume without rejecting seed 1's completed
receipt. Training arguments, checkpoint state, and the two-seed OOF gate remain
mandatory.

## Official-valid one-shot evaluation

After the OOF gate passes and both full-train checkpoints are verified, create
one persistent `O_CREAT|O_EXCL` attempt claim before enumerating valid image or
mask IDs or constructing the valid dataset. The claim is retained after any
failure, including concurrent invocation, and is not automatically removed;
a retry needs a new protocol decision. Per-image scores are written only to
the staged result files and are not printed during evaluation. The completed
output directory is published only after all 20 valid images and manifests
have been written. The evaluator still has no test-split option.

No Model392 training, OOF scores, official-valid scores, or test artifacts were
read when this addendum was written and locked. The existing Model382 job is
separate and continues under its own lock.
