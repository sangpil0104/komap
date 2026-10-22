# Source archive index

`historical_scripts/` preserves the 210 top-level Python sources from the experiment workspace. The authoritative mapping from model number to purpose, status, and recorded result is `docs/model_results.md`; best-score selection is summarized in `docs/best_models_by_miou.md`.

## Reproduction-oriented code

- `model322_team_b4/KoMaP/`: teammate EfficientNet-B4 recipe and grouped-CV utilities.
- `model326_hybrid_fullstats/KoMaP/`: full-image-stat input, training, prediction, grouped-OOF aggregation/evaluation, and per-image Lovasz loss used by Models 326/329/333.
- `model336_hrnet_best/`: HRNet-W48 + U-Net++ architecture, dataset, loss, inference, evaluator, and queue configuration.
- `model339_joint_gamma_route_search.py`: active CPU-only 400-setting joint-gamma route search; set `PYTHONPATH=experiments/historical_scripts` to import the archived shared route helpers.
- `runners/`: original P100 shell launchers/queue supervisors for Models 325, 326, 328, 329, and 333.

The route-search helpers in `historical_scripts/` include Models 181, 183, 184, 187, 198, 323, 335, 337, and 338. They consume cached probability maps and metadata that are intentionally not included; see the experiment log for required artifact paths and class-order conversions.
