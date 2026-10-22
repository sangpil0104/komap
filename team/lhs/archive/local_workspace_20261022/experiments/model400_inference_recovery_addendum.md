# Model400 inference recovery addendum

The first fixed inference attempt created the exclusive inference claim and
completed the first sorted image (`A117_step4_sample2`), then stopped while
formatting its progress message: the frozen evaluator accidentally passed
`flush=True` to `json.dumps` instead of `print`. No official-valid mask pixels
were decoded, no score was computed, and no result or inference manifest was
written.

Recovery is limited to completing the same locked HRNet seed-2 four-fold
probability maps. It preserves and hash-verifies the already completed map,
does not rerun or overwrite it, uses the unchanged original Model400 lock and
inference claim, and writes the original scorer's expected manifest only after
all 20 maps pass shape/normalization checks. It adds no calibration, route,
threshold, or other selection. The original one-shot score claim remains
unused; scoring is still allowed exactly once after recovery.

The recovery script and this addendum are themselves bound in a separate
recovery lock before any additional image inference. This does not amend the
candidate or original Model400 lock.
