# Model349 fusion805 lite20 B4ws — local retrained model

20 weights: R11 + E B4ws4 + HRNet4 + F1. Fixed stage2 result; no published score is claimed.
Training used S=0, all ancestor stages, all 12 E OOF branches, and four BN-frozen warmstarts.
Use conda komap. Verify all local hashes: `python -B verify_model.py`.

From this folder, fresh image-only inference:

```bash
python -B -u lite20/infer_compact.py --config lite20/selected_config_lite20.json --parent-package . --assets assets --input /path/to/images --output /new/output --gpus 2,3,4,5,6,7
```

Strict checks use this run's hashes; no --allow-retrained bypass is necessary.
