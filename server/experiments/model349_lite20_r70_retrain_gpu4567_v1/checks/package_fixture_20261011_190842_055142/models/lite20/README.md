# lite20 — local paired experiment

20 checkpoints; all SHA256 values verified, including newly trained R B4. Use conda komap.

```bash
python -B verify_model.py
python -B -u compact/infer_compact.py --config compact/config.json --parent-package . --assets assets --input /path/to/images --output /new/output --gpus 4,5,6,7
```
