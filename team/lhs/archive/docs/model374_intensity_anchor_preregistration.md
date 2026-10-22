# Al-matrix intensity anchor: train-only feasibility screen

> 한국어 안내: 이 파일은 평가 전 고정한 실험 protocol 원문을 보존하기 위해 영어로 유지합니다. 분석 결과와 현재 판정은 [모델 결과 기록](model_results.md)을 참고하십시오.

Status: protocol fixed before computing class-separation results. Official
validation labels and predictions are out of scope.

## Hypothesis

Model329 already uses raw gray, per-image full-image z-score, and per-image
empirical CDF channels. A global histogram mode may approximate the dominant
Al-matrix brightness, while robust image spread accounts for exposure/contrast.
The candidate is useful only if that unlabeled per-image anchor improves the
pooled Al3Ni-vs-Eutectic intensity separation beyond the existing CDF feature.
This is a low-cost proxy screen, not a claim that image intensity alone explains
segmentation errors.

## Fixed transformations

For each full grayscale image, define `mode` as the lowest intensity bin with
maximum frequency and `spread = max(q95 - q05, 1)` in 8-bit intensity units.
The candidate anchor channel is `(gray - mode) / spread`; it is computed from
the image alone and is not clipped. Controls are raw gray, Model329's exact
full-image z-score transform, and its exact full-image CDF transform. No
normalization parameters are fitted from masks, and no alternative percentiles,
mode rules, or thresholds are searched.

## Data and decision gates

- Use the 70 exactly-once grouped train-OOF images/masks (59 filename-proxy
  groups). The mask is used only to evaluate class-conditional intensity; the
  transform itself sees only the image.
- Report the fraction of modal-bin pixels labeled Al, image-macro Al3Ni vs
  Eutectic AUC, pooled AUC for raw/z/CDF/anchor, and the absolute gap between
  pooled and image-macro AUC. Compute paired specimen-group bootstrap intervals.
- Continue to an input-channel model experiment only if (i) the modal-bin Al
  fraction exceeds 50% in at least 70% of images, (ii) anchor reduces the
  pooled-vs-image-macro AUC gap by at least half relative to CDF, and (iii) the
  paired group-bootstrap 95% CI for anchor-minus-CDF pooled separability has a
  lower bound above zero and point gain is at least 0.02 AUC.
- If any gate fails, stop this direction without training. If all pass, design
  a separate four-fold grouped-OOF test before any GPU run; do not select or
  retune from official validation.

The score proxy is diagnostic only. A pass would justify testing a model, not
predict an mIoU gain.

## Result: no-go (2026-10-01)

The fixed screen used all 70 exactly-once grouped-OOF train images (59 filename-
proxy groups). The modal-bin premise was supported: median Al fraction at the
whole-image mode was `99.998%`, and 64/70 images had >50% Al pixels at that
mode. However, the proposed anchored transform failed the predictive proxy:

| Fixed intensity score | Pooled Al3Ni/Eutectic AUC | Image-macro AUC | Absolute gap |
|---|---:|---:|---:|
| Raw | 0.7992 | 0.8576 | 0.0584 |
| Full-image z-score (existing Model329 channel) | **0.8509** | 0.8576 | **0.0067** |
| Full-image CDF (existing Model329 channel) | 0.8375 | 0.8576 | 0.0201 |
| Al-mode anchor | 0.8310 | 0.8576 | 0.0266 |

Anchor minus CDF pooled AUC was `−0.0065` (group-bootstrap 95% CI
`[−0.0294,+0.0111]`); the anchor also widened rather than halved the CDF gap.
Only the modal-bin composition gate passed. The direction is stopped before any
GPU training. This diagnostic suggests the existing z-score is already more
effective than the tested anchor for this intensity-only proxy; it does not
prove a segmentation mIoU effect. Official validation remained unopened.

Reproduction: `experiments/model374_intensity_anchor_screen.py`; P100 JSON:
`/workspace/komap_model1/results/outputs/model374_intensity_anchor_screen/20261001_train_oof.json`.
