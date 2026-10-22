# Model349 lite20 — selected seed-search model

20 deployed checkpoints: R11 + E8 + F1. Fixed lite20 recipe; selected from 100 training jobs.
Validation-selected development score, not an independent generalization estimate.
Use conda komap. `python -B verify_model.py` checks all files.

```bash
python -B -u lite20/infer_compact.py --config lite20/selected_config_lite20.json --parent-package . --assets assets --input /path/to/images --output /new/output --gpus 4,5,6,7
```
