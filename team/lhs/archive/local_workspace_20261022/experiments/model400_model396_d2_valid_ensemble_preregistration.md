# Model400: Model396-D2 one-shot official-valid evaluation

Status: APPROVED_FOR_ONE_SHOT_VALID_EVALUATION

Decision: proceed with one fixed 20-image valid evaluation after the code/data
lock succeeds. This approval is from an independent Codex review; Claude was
not called and is not a prerequisite.

## Candidate and rationale

Model396's frozen train-only grouped-OOF screen passed its preregistered gates.
D2 is the equal probability mean of Model329 EfficientNet-B4 and Model382
HRNet-W48 seed 2: 79.9866% OOF, +0.8244 pp against B2 (group-bootstrap 95% CI
+0.4339 to +1.2615 pp; 4/4 folds positive). The four class deltas and Step4
delta were positive. This is selection-conditional grouped-OOF evidence, not
an official-valid result or a guarantee of an improvement.

The proposed valid route is not an exact replay of OOF D2. OOF used one
held-out fold checkpoint per family per image; valid inference will average
four fixed fold checkpoints per family. This is a reasonable, deterministic
final-inference route, but it is an estimand change and must be reported.
Model329's B4 valid probabilities are a pre-existing output from its fixed
four-fold prediction pipeline. The lock binds that archive, its score/console
receipts, the runner and prediction source, and all four checkpoint hashes;
the archived probabilities are not recomputed in this run. These receipts
support the recorded pipeline lineage but are not a fresh bitwise replay of
the B4 maps.

## Frozen prediction recipe

- Model329 B4: use its existing `valid_ensemble/probabilities.npz`, verified
  against the exact 20 valid image IDs, dimensions, four channels, normalized
  probabilities, score-receipt IDs, and locked source/checkpoint files.
- Model382: use only seed-2 folds 1–4. Verify saved arguments, each checkpoint
  hash, each fold manifest, and the Model392 sequence-closure references.
- Per HRNet fold: patch 512, input scale 2, target width 0, overlap 170,
  flip4 TTA, Gaussian tile-weighted probability stitching at original
  resolution, batch size 2. Average the four fold maps with equal weights.
- Final probabilities: `0.5 * Model329_B4_four_fold + 0.5 * Model382_seed2_four_fold`;
  argmax only after probability averaging. No threshold, calibration,
  postprocessing, router, per-image choice, or weight fitting.
- Score exactly once with image-wise mean of four class IoUs and empty union
  set to zero. Report all 20 image scores, four class means, overall mean, and
  delta from the recorded Model364 result. Do not read or use test data.

## Independent review and limitations

- The metric code uses per-image class IoU, averages the four classes per
  image, then averages images; empty class unions contribute zero. Palette and
  class order match both model families.
- Model396's preregistered OOF comparison provides the selection basis for the
  fixed 0.5/0.5 blend. It does not estimate the four-fold valid route exactly.
- Valid has 20 images and has been reused during development. Any Model400
  score is descriptive, selection-conditional development evidence—not an
  unbiased estimate of test/generalization performance.
- A score below 80% ends this valid look. Further choices return to
  train-only grouped OOF; no valid-based tuning is authorized by this record.
- No official-valid mask pixels were decoded while drafting/reviewing the
  protocol. The lock hashes mask files for split identity (so their bytes are
  read for SHA-256); HRNet inference uses `masks_dir=None` and must finish all
  20 maps before the scorer decodes labels.

## Fail-closed requirements

Before locking, verify the fixed protocol status, Model396 OOF lock/result and
all D2 gates, all model recipes/manifests/checkpoint hashes, Model329 archive
IDs/score-receipt IDs/source receipts, Model364 metric/value, exact 20 paired
image/mask IDs, inference/scoring sources, and runtime/dependency versions. The
lock binds file sizes and SHA-256 hashes. Refuse changed, missing, ambiguous,
or unexpected artifacts.

Inference must not decode masks. It must write exactly 20 finite, normalized,
shape-matched HRNet mean maps and an immutable manifest containing map hashes,
checkpoint hashes, configuration, and
`official_valid_mask_pixels_decoded_during_inference=false`. The scorer
verifies the complete manifest and every map before taking its exclusive
one-shot claim. It cannot overwrite or repeat a claim/result. Test files and
labels are excluded.

## Reviewer conclusion

GO for the one locked evaluation described above. The train-OOF support is
positive and the route is fully specified, while the reused-valid and
OOF-to-four-fold estimand limitations are explicitly disclosed. This is an
evaluation decision, not a claim that the candidate will reach 80%.
