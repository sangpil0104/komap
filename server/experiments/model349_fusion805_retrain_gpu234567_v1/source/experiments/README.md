# Experiment code map

This directory preserves the Python and shell sources behind the numbered KoMaP segmentation experiments. The model number is an experiment/result identifier, not a promise of one standalone source file per number: several numbers are settings or ensemble routes that reuse a trainer or scorer.

For the Model349 training bundle, provide external `train.zip` and `valid.zip`, then run `KOMAP_DATA_DIR=/path/to/archives bash experiments/run_model349_pipeline.sh train`. It trains the historical ancestor and component checkpoints, regenerates Model183 route entries, and searches the Model335→349 route chain. No old checkpoints, predictions, or competition images are included in the code ZIP. See `docs/model349_reproduction.md`.

## Where to look

- `historical_scripts/`: 210 legacy training, inference, audit, and parameter-search scripts retained under their original filenames. Shared imports from these files may require `PYTHONPATH=experiments/historical_scripts`.
- `model322_team_b4/KoMaP/`: the teammate EfficientNet-B4 recipe and its grouped cross-validation tools.
- `model326_hybrid_fullstats/KoMaP/`: full-image-stat B4 training, inference, and grouped-OOF code used by Models 326/329/333.
- `model336_hrnet_best/`: HRNet-W48/U-Net++ model, data pipeline, losses, training, prediction, and evaluation. Model347 reuses this trainer with train-only CE weights and a warm-start checkpoint.
- Top-level `modelNNN_*.py`: later standalone route searches, OOF audits, and exact metric recomputations. In particular, `model346_model336_into_341_route.py` is the shared exact scorer for Models 346/348/349/352/353; `model350_model346_fine_gamma.py` refines the Al/Eutectic-Si route coefficients with grouped-CV.
- `runners/`: recorded P100 launch/queue commands. Models 347→348→349→350 and 351→352→353 run in gated sequences; `runners/legacy/` contains provenance scripts with stale process/device assumptions and should not be launched unchanged.
- `../results/outputs/`: selected JSON configuration and score records. Large/non-source artifacts—including dataset archives, checkpoints, prediction masks, and cached probability maps—are intentionally excluded.

## Canonical records

- `../docs/model_results.md` is the chronological model-number → method/status/result record.
- `SOURCE_INDEX.md` is the short implementation map; `SOURCE_MANIFEST.md` lists every archived Python/shell source path (286 files: 249 Python, 37 shell).
- `../docs/best_models_by_miou.md` ranks models by metric policy. The official score is image-wise four-class mIoU with empty-union assigned 0; NaN-skip and pooled/global mIoU are reference metrics only.

## Reproduction limits

The archive contains source and selected metadata/results, not the private competition data, checkpoints, predictions, cached probability tensors, credentials, or a complete frozen environment. Many historical route searches consume those separately generated artifacts and are not standalone runs. Recent runner scripts use the recorded P100 paths `/workspace/komap_model1` and `/workspace/venv-komap`; review and adapt paths only in an authorized environment. No test mIoU is available because test labels are not provided.
