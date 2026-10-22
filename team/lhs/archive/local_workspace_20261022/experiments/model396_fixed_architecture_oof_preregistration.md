# Model396: fixed HRNet-W48 × EfficientNet-B4 train-OOF blend

Status: pre-result protocol, 2026-10-03 UTC. Do not inspect the combined candidate score until the input lock is created and verified.

## Question

Does fixed 1:1 probability averaging of the latest HRNet-W48/U-Net++ model family (Model382) and EfficientNet-B4/U-Net (Model329) improve held-out train image-wise mIoU more consistently than spending the same two-model budget on two Model382 seeds?

This is not routing: every image uses the same two-model arithmetic mean. It is CPU-only and uses only the 70 train images' exactly-once OOF predictions and train masks. The active Model392 run is not an input and must not be inspected or modified. Official validation and test images/labels are excluded.

## Frozen inputs and checks

- Metric: per image, compute IoU for Al, Al3Ni, Primary Si, and Eutectic Si; an empty union has IoU 0; average the four classes, then average images.
- Fixed fold masks: `data_model322_group_cv/oof_fold{1..4}/masks`.
- B4 probabilities: Model329 per-fold `fold{1..4}_oof/probabilities.npz`.
- HRNet probabilities: Model382 seed1/seed2 per-fold `fold{1..4}_oof/valid_probabilities.npz`, with the existing Model392 sequence-closure lock and each Model382 manifest/checkpoint reference verified.
- Every archive must match its fold's exact 70-image partition; each image and specimen group must be held out exactly once per model; all four probability channels must be in the official palette order `[Al, Al3Ni, Primary Si, Eutectic Si]`. Any mismatch closes the analysis as unestimable.
- Hash-lock this protocol, evaluator, masks, B4 args and OOF archives, Model382 sequence-closure receipt, and Model382 OOF manifests/probability archives before scoring. No file overwrite or missing-fold substitution.

## Fixed arms

- S1: Model382 seed1 alone.
- S2: Model382 seed2 alone.
- B2: equal mean of Model382 seeds 1 and 2 (two-model same-family control).
- D1: equal mean of Model382 seed1 and Model329 (two-model architecture-diverse candidate).
- D2: equal mean of Model382 seed2 and Model329 (two-model architecture-diverse candidate).

No weights, class biases, thresholds, TTA, routing, post-processing, or image-specific choices are tuned.

## Uncertainty and go/no-go rule

Use 20,000 paired bootstrap draws, seed 396, sampling specimen groups with replacement within each of the four fixed folds. All arms use identical draws. Report overall, four per-class, fold-wise, and Step4 image-mean mIoU changes in percentage points. Also report each specimen group's summed image-delta contribution.

Both D1 and D2 must satisfy every condition against B2: mean gain at least `+0.30 pp`, paired 95% CI lower bound above 0, and positive mean change in at least 3/4 folds. Each must also exceed its corresponding single-seed arm by at least `+0.50 pp`, with paired 95% CI lower bound above 0. Against B2, every class and Step4 must be no worse than `−0.50 pp`. For each D-vs-B2 comparison, the largest positive specimen-group summed contribution divided by the net summed gain must be at most 25%; non-positive net gain fails. A pass is only permission to consider a separately locked final candidate; it does not authorize tuning or constitute independent confirmation.

If any artifact, fold, channel-order, or metric check fails, report `UNESTIMABLE` without substituting valid/test data. If any numeric gate fails, report `NO_GO`; do not search other weights or routes.

## Interpretation limits

Model329, Model382, and related OOF summaries have already been examined, and the B4 recipe was developed after prior validation use. Any new blend result is development-only and selection-conditional, not an independent estimate of test performance. Do not open official valid or test for this analysis. The 80% objective remains; this screen alone cannot establish it.
