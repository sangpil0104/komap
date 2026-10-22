# Model407 predicted-component FP area audit (descriptive only)

## Motivation

The locked Model407 scorer reports unmatched 1–63 px component counts and total
target-class FP pixels. Its companion audit describes GT-component recall and
class confusion. Neither reveals whether FP pixel area is concentrated in
small zero-overlap islands, in small components that overlap GT, or in larger
components. This audit answers that diagnostic question without changing the
Model407 adoption gate.

## Fixed scope and measurements

- Run only after Model407's 70-image/59-group OOF artifact lock verifies.
- Read `train.zip` train masks and the fixed grouped-OOF probability maps for
  Model407 candidate, its matched control, and the already-locked Model392
  seed-1 parent. Never read official `valid.zip` or `test.zip`.
- Use native-resolution argmax predictions and 8-connected components for
  Al3Ni and Eutectic Si. Fixed native pixel bins are `[1,64)`, `[64,256)`, and
  `[256,+inf)`.
- For each image, class, and bin, report predicted component count, components
  with zero same-class GT overlap, components with nonzero overlap, and FP
  pixels partitioned between zero-overlap and overlapping components. Verify
  that the bins account for every target-class FP pixel.
- Also stratify every target-class diagnostic by whether that class is present
  in the image GT. Under the official empty-union=0 policy, an absent GT class
  receives IoU 0 whether its prediction is empty or nonempty, so its FP
  component burden is not a direct within-class IoU gain opportunity; changes
  may still affect the other classes through argmax competition.
- Report image-mean paired differences and 20,000 specimen-group bootstrap
  95% intervals for candidate vs control and candidate vs parent. Positive
  deltas mean a larger error/count burden. This is descriptive, not a score
  gate, threshold/blend search, or model-selection criterion.

## Integrity and stop conditions

- Verify Model407 source/calibration/OOF locks before reading probability
  maps. Record the OOF-lock SHA-256 and audit-script SHA-256 in the output.
- Write one new JSON output with exclusive-create semantics; refuse overwrite.
- Do not access official validation/test, modify the one-shot OOF score, or
  alter any adoption gate. Do not copy images, masks, or probability archives
  from P100 to the local workspace.
- If the lock or 70-image/59-group coverage check fails, stop without a
  partial report.
