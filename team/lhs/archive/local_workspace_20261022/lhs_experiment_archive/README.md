# KoMaP segmentation experiments

Private research archive for the KoMaP four-phase microstructure segmentation competition. It contains the experiment history and selected reproducibility code; it deliberately does **not** contain competition images/masks, dataset ZIP files, model checkpoints, prediction archives, or credentials.

## Official metric and latest recorded result

The official target is image-wise mean IoU: calculate the four-class IoU for each validation image, assign IoU `0` when a class has an empty union, average the four classes, then average images. The current raw validation best is **79.3683% (Model339)** on the 20-image validation set, so the 80% target is not yet met. Model339's paired group-CV versus Model338 does not support adopting it as a generalization improvement; Model338 remains the conservative route reference. NaN-skip and pooled/global mIoU are comparison metrics. Validation results are not test-set scores.

## Contents

- `docs/model_results.md`: chronological record of experiments 1–340, including completed, rejected, diagnostic, and currently running trials.
- `docs/best_models_by_miou.md`: concise current rankings under the official metric and reference metrics.
- `docs/miou_policy_audit.md` and its JSON files: historical metric-policy recalculation; the official empty-union=0 rule was confirmed later and is stated in the current summary.
- `experiments/historical_scripts/`: 210 top-level Python experiment and helper scripts, kept under their original filenames. For imports between archived scripts, set `PYTHONPATH` to this directory.
- `experiments/model322_team_b4/KoMaP/`, `experiments/model326_hybrid_fullstats/KoMaP/`: grouped-OOF training, inference, and evaluation implementations.
- `experiments/model336_hrnet_best/`: HRNet-W48 + U-Net++ experiment implementation and configuration.
- `experiments/model339_joint_gamma_route_search.py` and `model339_grouped_cv_parent_audit.py`: completed CPU-only joint class-wise route sweep and direct paired comparison with Model338; they import shared helpers from `experiments/historical_scripts/`.
- `experiments/model340_partial_oof_metric_audit.py`: exact empty=0 paired audit of the completed first two Model329 OOF folds versus Model322.
- `experiments/runners/`: P100 job scripts used for the recent grouped-OOF experiments. They document the original environment and may require path edits before reuse.
- `results/outputs/`: 584 JSON-only experiment records, scores, and run configurations copied from P100, preserving their output-directory layout. Model339 includes its exact route scores and paired group-CV audit; Model340 contains the two-fold interim paired OOF audit.

## Reproduction boundary

The training/evaluation code expects the competition dataset and generated checkpoints/probability archives to be supplied separately. Those artifacts remain on the authorized P100 workspace and are not versioned here. The JSON archive may contain metadata and historical absolute paths, but contains no image pixels, masks, probability arrays, or model weights. In the recorded environment, `KOMAP_WORKSPACE` referred to `/workspace/komap_model1` and `KOMAP_VENV` to `/workspace/venv-komap`; paths in the historical log are normalized to those variable names. Check each experiment's recorded command and dependency file before running it. Some historical scripts are one-off searches rather than standalone packages.

No test mIoU is reported: the competition test split has no labels available for local scoring.

## Provenance

The HRNet reproduction is based on the private `CNUKoMaP/HRNet_best` repository at commit `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`; related design notes came from `CNUKoMaP/sppark` at commit `e0d71b1`. See `docs/upstream_references.md`. The repository is private; keep the dataset and artifacts under the competition's access rules.
