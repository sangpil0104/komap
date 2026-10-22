# Model398: B4 matched fine-tune at two raster resize/crop scales

Status: conditional follow-up protocol drafted 2026-10-03, before opening Model392 or Model397 scores. This does not alter either locked experiment and authorizes no concurrent run. Claude Code Opus conditionally endorsed the matched-control design; its metric recommendation was corrected here to the competition's image-wise metric.

## Question and interpretation

Does changing only the per-image resize target from 4000 to 7680 during a short, matched B4 fine-tune improve held-out train OOF segmentation over an otherwise identical fine-tune at target width 4000?

This tests a **raster resize/crop-scale policy**, not calibrated physical magnification: no micrometre-per-pixel calibration is assumed. With a fixed 1536-pixel crop, target width 4000 presents about 38.4% of each image width; target width 7680 presents 20.0%. This changes the apparent size of phases and the field of view together. It does not isolate optical resolution from context, and upsampling cannot create source detail. The target-width-dependent image-global/zcdf3 preprocessing is part of this practical scale-policy treatment and must be identical to its normal pipeline in both arms.

The protocol is conditional on Model392 and the already-registered Model397 sequence finishing first. It may be run only if none of those pre-fixed candidates reaches 80.00% on its authorized official-valid evaluation. If a fixed candidate reaches the target, do not launch Model398. Any OOF conclusions remain development evidence conditional on extensive prior use of these train groups; valid results remain descriptive on the repeatedly used 20-image set. Test stays sealed.

## Fixed data and metric

- Use the existing four specimen-group folds and all 70 train images / 59 specimen groups. Verify each image has exactly one held-out prediction and no specimen group crosses folds.
- Primary endpoint is the official image-wise four-class mIoU: calculate each class IoU per image, assign zero to an empty union, average the four classes per image, then average over images. Report percentage-point paired deltas. Do not substitute pooled-confusion mIoU.
- Report each class's image-mean IoU and Step4 image-wise mIoU as safety endpoints. Group-bootstrap 20,000 paired draws, resampling groups within each fixed fold, seed 398.
- For mechanism context only, report Model387's fixed native-resolution phase-boundary distance bands (`<2`, `2–<4`, `4–<8`, `>=8` pixels) for Al3Ni and Primary Si, including the `<4px` false-negative rate. These are secondary diagnostics, not alternative scores or tuning targets.
- Official valid masks, maps, or scores are not read during training or OOF screening. No threshold, TTA, post-processing, checkpoint, or weight search.

## Matched intervention

For each fold, initialize both branches from that same fold's frozen Model329 `last.pt`. Keep the same B4 architecture, data split, seed, update count, augmentation, optimizer schedule, and inference procedure. Change only `target_width`:

| Setting | Matched control | Treatment |
|---|---:|---:|
| `target_width` | 4000 | 7680 |
| Architecture / encoder | U-Net / EfficientNet-B4 | same |
| Input / normalization | `zcdf3` / `image_global` | same |
| Crop / sampler / batch | 1536 / uniform / 2 | same |
| Fine-tune updates | 2000 | 2000 |
| LR / weight decay | `2e-5` / `1e-4` | same |
| Dice / Lovasz / clDice weights | `1 / 0 / 0` | same |
| Initialization | same fold's Model329 checkpoint | same checkpoint |

Use `--no_valid_eval`, `--deterministic_workers`, four workers, AMP, fresh identical AdamW/cosine scheduler settings, and fixed final-iteration checkpoints. Use the same fine-tune seed within each matched pair; the changed raster scale can still change crop coordinates, so this is not pixelwise-paired augmentation. Do not resume the parent optimizer state. Confirm strict checkpoint loading and that preprocessing metadata records the assigned target width.

Generate three exactly-once OOF prediction arms on each corresponding fixed held-out fold, all at native label resolution:

- `P0`: the untouched Model329 fold checkpoint, re-inferred at target width 4000;
- `C`: the 2000-update target-width-4000 fine-tune;
- `T`: the 2000-update target-width-7680 fine-tune.

Model329's existing `common.predict_probs` is full-frame D4 inference, not tiled inference. Do not compare a tiled `C`/`T` map to the archived full-frame Model329 OOF map as the paired parent endpoint. Recompute `P0`, `C`, and `T` with the same locked overlap-tile procedure: resize the grayscale image to the arm's target width; compute `image_global`/`zcdf3` input statistics and CDF from that complete resized image using Model329's preprocessing; cut normalized input into 1536 x 1536 tiles with overlap 384; apply D4 TTA; blend tile probabilities with the fixed Gaussian window; then area-downsample probabilities to the original label grid. Use batch size 1 and `input_scale=1`. No validation split is passed to the trainer.

The generic `model336_hrnet_widthnorm_candidate/komap_seg/inference.py` provides the overlap/Gaussian/D4 mechanics but expects an already normalized tensor. Add a B4 adapter that performs Model329's target-width resize and whole-resized-image `make_input` preprocessing before calling it. Freeze and smoke-test the adapter, including output shape, finite normalized probabilities, and a small synthetic tiling case, before any Model398 training or score access. The exact same adapter/settings must be used for P0, C, and T; the historical full-frame Model329 OOF score is context only, not a gate comparator.

## Sequential screen and confirmation

### Seed-0 screen

Run four control and four treatment fine-tunes: 16,000 total optimizer updates. Compare treatment (T0) with matched control (C0), and also compare T0 with `P0`, the corresponding untouched Model329 checkpoints re-inferred at target width 4000 using the same locked tiled procedure.

Proceed to the seed-1 replication only if all screen conditions pass:

1. T0−C0 official image-wise mIoU is at least `+0.75 pp`, its paired group-bootstrap 95% CI lower bound is above zero, and at least 3/4 fold deltas are positive.
2. No per-class or Step4 T0−C0 image-mean IoU delta is below `−0.50 pp`.
3. T0−P0 image-wise mIoU is at least `−0.25 pp`; this guards against a large degradation from the starting model while treating parent OOF as selection-conditional.
4. Neither Al3Ni nor Primary-Si `<4px` boundary-band FN rate worsens by more than 2 percentage points versus C0. This is a safety check, not a claim that these bands represent physical objects.

If these conditions fail, stop Model398. A weak/null one-seed screen is recorded as `SCREEN_NOT_PASSED`, not proof that scale can never help. Do not change the 2000-update budget after seeing OOF scores.

### Seed-1 confirmation

Only after the screen passes, repeat both matched branches on all four folds from the original Model329 fold checkpoints with seed 1: another 16,000 updates. Do not initialize from seed-0 fine-tunes.

Average seed-0/seed-1 probabilities within each arm before argmax. The final scale-screen gate requires all of:

1. T−C official image-wise mIoU is at least `+0.50 pp`, group-bootstrap 95% CI lower bound above zero, and at least 3/4 fold deltas positive.
2. Both seed-specific T−C point deltas are positive.
3. T−P0 is at least `+0.25 pp` with paired group-bootstrap 95% CI lower bound above zero.
4. Every class and Step4 T−C image-mean IoU delta is at least `−0.50 pp`; both thin-class `<4px` FN rates satisfy the same 2pp non-worsening check.

Only a full confirmation pass permits one fixed official-valid evaluation of the four-fold treatment ensemble. The valid scorer and its input lock must be frozen and smoke-tested before opening valid masks. Do not score the matched-control arm on official valid or tune from any valid result. Report the result beside the earlier fixed Model392/Model397 looks and label it reused-validation development evidence. No test inputs or labels are allowed.

## Compute, failure modes, and provenance

- Seed-0 screen costs 16,000 updates; confirmation, if earned, costs another 16,000. Each training crop tensor remains 1536², so model step cost should be similar; image preprocessing changes. At fixed 1536 tile and 384 overlap, a 7680-wide image requires about 1.92x as many horizontal and vertical tiles as a 4000-wide image; for similar aspect ratios this is about 3.7x the total tiles, and D4 multiplies tile forwards by eight. Measure actual wall time on representative data; do not promise a score gain. Full-frame inference at width 7680 is not assumed to fit P100 memory.
- Main risks: interpolation without new source detail; reduced field of view; altered zcdf3/image-global statistics caused by resized-image histograms; one-seed screen uncertainty; group-bootstrap intervals omitting model-training variance; and historical OOF selection bias.
- Before any training, lock this protocol, runner/evaluator/smoke source, Model329 fold args/checkpoints/OOF archives, exact fold manifests and train masks, preprocessing dependencies, and fixed Model392/397 decision artifacts. Every output path is new and non-overwriting. A later valid-stage lock is separate and gated.

## Implementation feasibility check

The existing `model326_hybrid_fullstats/KoMaP/train.py` exposes `--init_checkpoint`, `--target_width`, `--no_valid_eval`, and the required B4 input/loss arguments. This is a static code inspection, not a run. The shared B4 adapter [model398_b4_tiled_inference.py](model398_b4_tiled_inference.py) now applies whole-resized-image Model329 preprocessing and the locked tile/D4 settings; its CPU-only synthetic smoke [model398_b4_tiled_inference_smoke.py](model398_b4_tiled_inference_smoke.py) passed. No real image, checkpoint, score, or GPU was used by that smoke. Before assigning a Model398 result, still add and lock (1) a runner that writes separate control/treatment fold outputs and (2) an evaluator that enforces exactly-once group OOF and the official image-wise metric above for P0/C/T.
