# Model409 OOF-output exposure addendum

Date: 2026-10-06. This addendum preserves, rather than edits, the v2
preregistration and source/calibration locks.

At 04:17 UTC, a status command tailed `runner.log` just after fold1 held-out
inference finished. The terminal portion included a per-fold class-score
line, which appeared in tool output. The value is not copied into this file,
recomputed, or used for a decision. No official `valid.zip` or `test.zip`
data were read. The exposure occurred before folds 2–4 were trained/scored.

The locked Model409 candidate/control recipe, folds, inference, and one-shot
four-fold scorer remain unchanged and will be allowed to finish. From this
point, progress checks use only process state and file metadata; do not read
the runner/evaluator logs or fold-level score summaries. The final OOF result
must disclose this partial metric exposure and is exploratory/selection-
conditional only: it cannot pass an adoption gate, authorize seed-2 or
official-valid evaluation, or be presented as confirmatory evidence. No
hyperparameter or treatment change may be based on the exposed line.

The standalone Model410 weighted-CE protocol draft was written before this
exposure. It remains an unapproved draft and cannot use Model409's eventual
exploratory result to tune weights or correction strength.
