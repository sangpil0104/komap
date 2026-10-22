# Source archive index

`historical_scripts/` preserves the 210 top-level Python sources from the experiment workspace. The authoritative mapping from model number to purpose, status, and recorded result is `docs/model_results.md`; best-score selection is summarized in `docs/best_models_by_miou.md`.

Model numbers identify experiment outcomes and route variants, not necessarily unique architectures or files. Training variants may share a trainer, while ensemble/calibration models may be defined by a search script and saved configuration. The 2026-09-30 audit found 286 Python/shell sources (249 `.py`, 37 `.sh`), all tracked in this archive; the exhaustive file list is in `SOURCE_MANIFEST.md`. Every top-level Python/shell source in the experiment workspace had a corresponding archived source. Where copies differed, the archive retains the later corrected version (including explicit B4 channel-order handling and Model336 weighted-CE warm-start support), so do not replace archive files with older workspace copies. Four previously omitted launchers are preserved under `runners/legacy/`.

For an exhaustive path-by-path list of the archived Python and shell files, see [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

## Find code for a model number

The model number is a result/experiment ID, so one source can implement several numbered variants. Use `docs/model_results.md` for the exact method, status, and score of each number, then follow its code links. For a quick starting point:

| Model IDs / family | Main implementation location | Notes |
|---|---|---|
| 1–321 | `historical_scripts/` plus `runners/` | Legacy training, inference, calibration, route search, and audit scripts retain their archived filenames. The result log is the model-number index. |
| 322 | `model322_team_b4/KoMaP/` | Teammate EfficientNet-B4 training recipe and grouped-OOF tools. |
| 323–335 | `historical_scripts/` plus `runners/` | B4 reproduction, route fusion, grouped-OOF, and metric audits; several numbered results share evaluators. |
| 336–337 | `model336_hrnet_best/` plus `historical_scripts/` | HRNet-W48/U-Net++ trainer and inference, then route-search helpers. |
| 338 | `historical_scripts/model338_fine_gamma_route_search.py` | Fine-gamma route search. |
| 339–346 | numbered sources directly under `experiments/` | Joint/fine route search, OOF-fitted blends, and exact scoring; shared source files are linked from the corresponding result entries. |
| 347–349 | `model336_hrnet_best/` + `model346_model336_into_341_route.py` + `runners/` | Weighted-CE HRNet fine-tune and two route variants sharing the exact scorer. |
| 350–353 | `model350_model346_fine_gamma.py`, `model336_hrnet_best/`, `model346_model336_into_341_route.py`, and `runners/` | Fine-gamma search, D4 TTA inference, D4 route search, and Model351 D4 expert added to the fixed Model349 route. |
| 354–364 | `model354_model349_bias_audit.py`, `model355_model349_spatial_audit.py`, `model360_entropy_gated_d4_into_355.py`, `model364_model361_entropy_into_360.py`, and `../scripts/` | Spatial/entropy route continuation, scale-3 HRNet fine-tune helpers, and Model364 scoring. |

This inventory does not mean every historical run is independently reproducible from source alone. Some entries are route settings or checkpoint evaluations; their weights, cached probability maps, private data, and parts of the original runtime environment are intentionally not included.

## Reproduction-oriented code

- `model322_team_b4/KoMaP/`: teammate EfficientNet-B4 recipe and grouped-CV utilities.
- `model326_hybrid_fullstats/KoMaP/`: full-image-stat input, training, prediction, grouped-OOF aggregation/evaluation, and per-image Lovasz loss used by Models 326/329/333.
- `model336_hrnet_best/`: HRNet-W48 + U-Net++ architecture, dataset, loss, inference, evaluator, and queue configuration.
- `model339_joint_gamma_route_search.py`: completed CPU-only 400-setting joint-gamma route search; set `PYTHONPATH=experiments/historical_scripts` to import the archived shared route helpers.
- `model339_grouped_cv_parent_audit.py`: recomputes the paired leave-one-specimen-group-out comparison of Model339 route selection against the fixed Model338 route.
- `model340_partial_oof_metric_audit.py`: recalculates exact image-wise mIoU with empty=0 from paired saved masks for currently completed Model329 OOF folds.
- `model341_model329_joint_route_search.py`: CPU-only 626-setting class-wise Model339/329 route search, including Model338 as a fixed reference.
- `model342_oof_fitted_b4_blend.py`: fits a class-wise Model322/329 probability blend using train-only grouped-OOF predictions, then scores the selected blend on validation.
- `model343_model342_into_341_route.py`: completed 625-setting Model342-on-Model341 route probe; it selected the unchanged Model341 route.
- `model333_partial_oof_audit.py`: exact empty=0 paired OOF audit for Model333's CE+Dice control and Lovasz arms; the control-vs-parent and corrected Lovasz comparison cover all 70 train images. The paired evaluator accepts explicit parent/candidate channel orders to prevent class permutation mistakes.
- `model344_model333_control_blend_search.py`: searches a class-wise Model322-parent/Model333-control probability blend using train grouped-OOF predictions, then supports fixed-gamma validation scoring.
- `model345_model344_into_341_route.py`: completed CPU-only 625-setting comparison of the fixed Model344 expert against Model341; original and four-worker rerun JSON records are archived, all image-mean and class-score arrays match exactly, and no route beats the unchanged baseline.
- `model346_model336_into_341_route.py`: shared CPU exact-route implementation for Model346/348/349/352/353. Model346 compares the final Model336 HRNet expert against Model341; it verifies every valid target, class order, probability normalization, and baseline parity before accepting results. The selected route scores 79.8940% strict image-wise valid mIoU; its grouped-CV bootstrap CI narrowly includes zero.
- `model346_model336_into_341_route.py --expert-id 347`: Model348 mode, comparing weighted-CE Model347 probabilities against Model341 with grouped-CV and strict baseline parity.
- `model346_model336_into_341_route.py --expert-id 347 --anchor-model346`: Model349 mode, fixing Model346's selected gamma before adding Model347; baseline parity checks the fixed Model346 result before running the 625-candidate grid.
- `model350_model346_fine_gamma.py`: 315-route Model336 sweep over Al gamma `0.65–1.0` and Eutectic Si gamma `0–0.5`, both at `0.025` intervals; Model346 route parity, grouped-CV, and bootstrap uncertainty are included.
- `model336_hrnet_best/komap_seg/inference.py`: default-compatible flip4 inference plus opt-in 8-way dihedral probability TTA; all rotations/reflections are inverted to original image coordinates before averaging.
- `model346_model336_into_341_route.py --experiment-id 352 --expert-id 336`: repeated 625-route search with Model351's D4 probabilities; the experiment ID changes result labeling only.
- `model346_model336_into_341_route.py --expert-id 351 --anchor-model349`: fixes Model349's selected Model346+Model347 route, verifies exact Model349 score/class-mean parity, then tests Model351 D4 probabilities as an additional class-wise expert with grouped-CV uncertainty.
- `model354_model349_bias_audit.py` and `model355_model349_spatial_audit.py`: Model349 spatial diagnostic and fixed spatial route implementation used by Model360/364.
- `model360_entropy_gated_d4_into_355.py`: fixed entropy-gated Model351 D4 probability branch on Model355.
- `model364_model361_entropy_into_360.py`: original Model364 grid/scoring entrypoint; it verifies Model360 pixel-mask and score parity before comparing Model361 entropy-gate settings.
- `../scripts/train_model336.sh`, `train_model347.sh`, `train_model361.sh`, and `infer_model351_d4_valid.sh`: configurable project-root launchers for the HRNet parent/branches and D4 inference. They refuse to overwrite existing outputs.
- `../scripts/run_model364_valid_reproduction.sh`: fail-fast input check followed by the original Model364 scorer. It reruns the historical validation grid and therefore must not be interpreted as an independent test estimate.
- `model336_hrnet_best/train.py` + `komap_seg/losses.py`: preserve the unweighted Model336 default while exposing optional CE class weights and a model-checkpoint warm start for controlled follow-up fine-tunes.
- `runners/`: original P100 shell launchers/queue supervisors for Models 325, 326, 328, 329, 333, 341, 342, 343, 344, 345, 346, sequential weighted-CE HRNet/route Models 347–349, Model350 fine-gamma, Model351 D4 inference, and Model352/353 routing.
- `runners/legacy/`: earlier Model326 queue/metric-aligned runner and backbone recovery/sequential launchers for Models 82–90. These are provenance artifacts, not ready-to-run commands: the recovery scripts wait on historical `/proc` PIDs and select GPU 3.

The route-search helpers in `historical_scripts/` include Models 181, 183, 184, 187, 198, 323, 335, 337, and 338. They consume cached probability maps and metadata that are intentionally not included; see the experiment log for required artifact paths and class-order conversions.
