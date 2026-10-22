# Model382 two-seed mean sensitivity v2 addendum

This addendum strengthens provenance checks for the already frozen secondary
analysis in `model382_two_seed_mean_sensitivity_preregistration.md`. It does
not amend Model382's primary comparison, the v1 metric, the equal 1:1
probability mean, or any adoption threshold.

Before seed2 OOF files exist, the v2 receipt must freeze SHA-256 for every one
of the 70 fixed train-OOF ground-truth masks. It also records the exact mask
inventory (70 images, 59 specimen groups, fold assignment, Step4 coverage,
and whether A71 is present). Evaluation must reproduce this inventory and
every hash before opening any probability archive. If A71 is absent—as it is
in the fixed train-OOF inventory—the A71-excluded sensitivity is not an
independent subset result and must be identified as equal to the all-image
result.

After masks pass, evaluation verifies all eight seed/fold checkpoint, manifest,
and probability archive triplets with the existing Model382 verifier. Each
seed2 artifact must postdate the v2 receipt, and corresponding seed1/seed2
checkpoint and probability-archive hashes must differ. A failed provenance
check aborts before scoring; it is never repaired by substituting valid/test
data or choosing another artifact.

The resulting report includes source hashes and minimal runtime provenance
(UTC time, hostname, Python, NumPy, Pillow, and PyTorch versions where
available). The analysis remains development-only, CPU-computed, grouped
train-OOF evidence. Official validation/test inputs are not accepted.

The v2 synthetic smoke test additionally checks that a positive overall
image-mIoU delta cannot hide a greater-than-1pp class regression. For example,
three classes gaining 2pp and one class losing 3pp yields an overall +0.75pp
but must fail the class-safety gate.
