# lite20 r70 + HR336 seed 3411

17 checkpoints: full-train70 B4 322/326/329; only HR336 differs from the r70 package.
Verify: `python -B verify_model.py` in conda komap.
Inference: `python -B compact/infer_compact.py --config compact/config.json --parent-package . --assets assets --input /path/to/images --output /new/output --gpus 4,5,6,7`
