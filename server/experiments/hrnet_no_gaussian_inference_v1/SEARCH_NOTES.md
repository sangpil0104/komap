# Search decisions — 2026-09-29

- Use the frozen final HRNet_taeho_no_gaussian checkpoint, never retrain.
- Score official valid20 with empty class IoU=0; the historical report used 1.
- A00 batch4 differs from historical batch8 by only -0.0001012022 percentage points. All A/B conditions use batch4.
- A01 area and A00 bilinear are identical at the exact 2x reduction used here.
- First completed group: bicubic restoration +0.017598 pp; no TTA -0.343706 pp. Spatial merge weights have small effects.
- Input bilinear and antialiased bicubic hurt (-1.189232 and -0.520367 pp versus A00). Preserve original input bicubic in subsequent combinations unless new evidence supports another setting.
- Blurring predictions, grayscale-guided smoothing, tiny-component removal, and confidence-gated majority voting all hurt on the first postprocessing grid.
- The implementation ID `median_confidence` denotes per-class majority voting gated by probability margin; it does not use numeric class-ID median as the decision.
- Class1 upweighting and class3 downweighting have small favorable effects; Q follow-ups expand those ranges.
- All hyperparameter choices are validation-guided exploratory findings, not independent test gains.

## 13:04 UTC update

- Larger context at unchanged 2x magnification improves the original baseline modestly. With relative overlap held approximately fixed and batch1, native 160/192/224 score 78.522/78.741/78.872%, versus native 256 at 78.917%; native 512 and 768 reach 79.015/79.038%.
- Holding native context=256 and overlap=170 fixed, input magnification 1/1.5/1.75/2.25/2.5/3 scores 69.995/76.273/77.656/78.413/78.084/76.130%. Magnification and context must be treated as separate factors.
- Size-dependent magnification policies K00–K05 all decrease the valid20 mean. Do not infer that simply enlarging small images solves their boundary errors.
- C640_d4_bicubic reaches 79.093668% without class weighting. Its LP01 variant (Al3Ni probability ×1.2, then normalize) reaches 79.125377%. At this intermediate point, mean change is +0.481235 pp over nine images of width≤800 and -0.015210 pp over eleven larger images. This is descriptive post-hoc stratification, not a rule used by inference.
- Grayscale-guided local probability messages BM00–BM08 all underperform the baseline (78.618–78.900%). This negative branch is closed; do not label it as a successful boundary fix or a full DenseCRF implementation.
- Full-image FP32 conditions that ran out of memory are excluded unless all20 finish. A follow-up retries with expandable allocator segments at the same 36% per-GPU memory cap, including a matched batch1 control. BF16 full-image inference completed at 79.025508%.
- Follow-up core factors independently vary gamma, reflection halo, window, and averaging. Additional interpolation tests separate kernel library and align_corners. The new resampling engine reproduces the prior implementation exactly on a synthetic full tiling/TTA test; a real valid20 equivalence control is also scheduled.
- The 1e-6 constant-image check exposed a 1.192e-6 FP32 error in native torch bicubic align_corners=True. The engine matches torch bit for bit; the documented constant tolerance is 2e-6. Probability validity tests pass.
- Original checkpoint/source/report preservation checks pass; inference is separate from the user-started B4 training, which has naturally advanced to T01_native_geometry.
