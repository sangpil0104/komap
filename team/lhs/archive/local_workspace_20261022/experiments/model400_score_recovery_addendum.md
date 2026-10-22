# Model400 score-execution recovery addendum

The first score command passed the immutable Model400 lock and inference-map
preflight, created the original scoring claim, decoded the first valid mask,
and computed that image's class IoUs in memory. It then stopped at the
progress message because the frozen evaluator passes `flush=True` to
`json.dumps` rather than `print`. No numeric metric was printed, no aggregate
was computed, and no Model400 result file was written. The original score
claim is retained as an audit record.

One recovery execution is authorized solely to complete this same fixed
score. It must use the same locked probabilities, labels, class order,
image-wise four-class IoU, empty-union=0 policy, and no test inputs or tuning.
The recovery runner applies a narrow JSON logging compatibility shim and uses
a distinct exclusive recovery claim; the original claim is never removed or
overwritten. If a result exists or the recovery claim has already been taken,
the runner refuses to proceed. The recovery runner and this addendum are
hashed in a separate lock before it decodes any further labels.

This is completion of the interrupted Model400 measurement, not a new model
candidate or an additional validation-based selection. The final result must
be labeled selection-conditional and descriptive because the 20-image valid
set has been reused during development.
