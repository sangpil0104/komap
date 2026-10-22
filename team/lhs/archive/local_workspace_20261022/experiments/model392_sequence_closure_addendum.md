# Model392 sequence-closure addendum (2026-10-03)

This is a supplemental execution-sequence decision for the already frozen
Model392 revision-3 protocol. It does not edit or supersede the Model392
candidate, evaluator, numerical gates, fold geometry rule, training recipe,
inference options, full-train policy, or one-shot official-valid policy. The
revision-3 runtime lock remains unchanged. A separate sequence-closure receipt
binds this addendum and the exact Model382 reference artifacts before any
Model392 run starts.

## Close the unavailable prerequisite without substituting data

The previous sequence required the Model382 matched-count comparison and the
Model387/388/389 locked audits to finish before another GPU experiment. The
Model382 seed-count/two-seed analysis and locked audits have been run, but the
main matched-count B/C contrast is not estimable: the required Model374
folds3–4 train-OOF probability archives and Model376/377 OOF archives are not
available in the recorded P100 outputs. The missing Model374 outputs were
rechecked at their documented Drive destination. The metadata-only inventory
([receipt](results/model374_drive_metadata_inventory.json)) finds completed
epoch-70 checkpoint pairs for folds3–4, but each Drive `oof/` directory lists
`valid_*` artifacts rather than a train-fold OOF archive. File contents were
not opened. The existing Colab session name is no longer found by the CLI.

The unavailable B/C result is therefore formally closed as **unestimable after
the recorded recovery search**. Do not substitute official valid/test results,
and do not retrain old arms as if their original stochastic runs had been
recovered. The fold3–4 Model374 checkpoints are not substitutes for their
missing OOF predictions. The existing Model382 reference artifacts for both
seeds and all four folds remain available for the already preregistered
same-seed/fold Model392 comparison.

The terminal audit sequence has no GPU branch awaiting execution: Model387's
locked audit is complete; Model388's prospective sampler branch did not pass
the realized-dose gate (Model395 CPU audit, NO-GO); and Model389's rare-
component miss-rate gate was NO-GO. No sampler branch is authorized by those
results. Their existing reports and gates remain unchanged.

## Decision and inference status

Proceed with Model392 revision 3 only: two seeds × four fixed group folds,
executed sequentially on the single P100. Keep Model390 separate and do not
mix its RNG intervention into this run. The Model392 candidate output tree
must be absent before the closure lock is created.

Model382 aggregate and partial per-image OOF results were exposed in prior
development logs before this sequence-closure decision; several later
train-only audits also reused those groups. Model392 itself had no training,
OOF result, official-valid result, or test result at closure time. Its recipe
and numeric gates were locked before the candidate outcomes, but the eventual
comparison is still development evidence conditional on prior exposure and
experiment selection—not an independent estimate of generalization. Report
this caveat prominently; do not use the result as unbiased confirmation.

All eight Model392 runs and exactly-once held-out-train OOF inference must
finish before the confirmatory evaluator is invoked. Per-fold inference logs
may contain OOF scores and must remain uninspected until all eight fold
manifests exist. No intermediate candidate scores may change the schedule,
recipe, or gates. The evaluator's fixed decision rule remains exactly as in
the revision-3 lock:

- Each seed's paired image-mean delta versus the same-seed/fold Model382
  reference is positive.
- The equal-weight two-seed delta is at least `+0.50%p`, with paired,
  fold-stratified 59-group bootstrap 95% CI lower bound above zero.
- For each seed and pooled result, at least 3/4 folds improve, the worst fold
  is at least `−0.30%p`, every class is at least `−0.50%p`, and Step4 is at
  least `−0.50%p`.

If every condition passes, the already locked sequential full-train seed1/2
workflow and exactly-once official-valid 20-image evaluation may proceed. The
official-valid evaluation is not a tuning loop; no threshold, scale, TTA,
weight, or postprocess is selected from it. If the OOF gate fails, stop Model392
without full-train or official-valid evaluation. Test remains unopened either
way. Passing the OOF gate or reaching 80% on valid is not guaranteed by this
addendum.

The supplementary lock must record the addendum hash, revision-3 lock hash,
the Model382 pre-result lock hash, hashes/sizes/timestamps for each seed/fold
Model382 checkpoint, args file, OOF manifest and OOF probability archive, and
the four terminal audit result hashes. The helper verifies the existing locks
and Model382 manifests but does not decompress or score any OOF archive. OOF
archive bytes are read only to calculate their SHA-256. It refuses to overwrite
an existing closure lock or rebind a run receipt to a different closure; for a
matching run receipt it atomically adds the closure/addendum hashes. The P100
runner rechecks the train/held-out symlinks immediately before inference, rejects
symlinked output paths, and recovers a missing closure binding after an
interruption without deleting stale temporary files. No official-valid or test
file is part of the closure lock inputs.
