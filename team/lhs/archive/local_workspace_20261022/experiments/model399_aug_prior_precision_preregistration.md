# Model399 crop-prior augmentation precision audit

Status: pre-score, training-data-only diagnostic. This is not Model399 training, not an OOF mIoU analysis, and does not authorize a class-weight recipe.

## Question

Does extending the actual Albumentations crop-prior probe from one epoch (624 crops per fold/seed) to ten epochs materially change or stabilize the estimated post-augmentation class prior `q` used by the proposed sampler-aware `1/q` recipe?

## Frozen procedure

- Re-run the existing `experiments/model399_crop_prior_audit.py` on folds `1,2,3,4`, coordinate-cap seeds `1,2`, `native_epochs=70`, and `aug_probe_epochs=10`.
- Keep the crop rule, crop RNG, source-prior definitions, class order `[Al, Al3Ni, Primary Si, Eutectic Si]`, model code, data split, and all analysis formulas unchanged. The only change from the existing audit is ten repeated epochs through its existing training augmentation transform.
- Run on P100-hosted CPU with `CUDA_VISIBLE_DEVICES=""` and the existing `/workspace/venv-komap` environment. The audit source is SHA-256 `2a96b0dfaba36db1cd9fbbabbf683fe6a63da75db80ca8540a7ee18a30ca3ddf`; the one-epoch reference result is SHA-256 `696c322f26917d45db49d9ab0054a688f1d1e1372f2e86ba7a2b82593e0bc052`.
- Write only to the new, previously absent path `experiments/results/model399_crop_prior_audit_aug10.json`; the script uses exclusive-create mode. Verify all fold training-mask hashes exactly match the one-epoch reference before comparing `q` or inverse weights.

## Allowed interpretation

Report the ten-epoch `q` means, normalized fold/seed inverse-weight ranges, and differences from the existing one-epoch estimate. Any change in `q` is a Monte Carlo/augmentation-prior estimate, not a performance result. Do not compute mIoU, inspect any OOF predictions/labels, touch official valid/test, train a model, select a weight/correction, or alter Model399 gates. The diagnostic may only inform Claude's later choice between static source-prior and sampler-aware recipes; both remain unapproved until that review.
