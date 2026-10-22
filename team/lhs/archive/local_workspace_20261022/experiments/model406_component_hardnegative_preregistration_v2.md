# Model406 v2: low-memory small-component correction calibration

This is a calibration-execution revision only; it does not change the candidate loss, parent checkpoints, training recipe, fixed grouped folds, or adoption gates in `model406_component_hardnegative_preregistration_v1.md`.

## Calibration revision

- The v1 calibration did not complete because its batch-of-four gradient probe exhausted available device memory. It made zero optimizer steps and produced no candidate/control checkpoints.
- v2 processes one crop at a time, for 32 outer-train crops per fold. This preserves the v1 probe's total crop count (8 batches × 4 crops) while lowering peak activation memory.
- The locked statistic is the median of per-crop joint-auxiliary/main decoder-gradient norm ratios. The coefficient remains `0.10 / median_ratio`. This differs from a batch-of-four gradient ratio and is declared up front; no held-out data or mIoU enters calibration.
- Only masks/images in the fixed outer-training split are read. At least 24 of 32 crops must contain a targeted component in each fold. No optimizer step is permitted.

All other candidate/control training and grouped-OOF evaluation decisions inherit the v1 preregistration unchanged. Official valid and test remain sealed.
