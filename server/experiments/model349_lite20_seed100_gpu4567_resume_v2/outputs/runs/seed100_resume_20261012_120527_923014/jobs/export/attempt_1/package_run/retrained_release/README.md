# Model349 lite20 R — newly retrained weights

All 11 weights were retrained. The lite20 inference code, calibration, fold subset and fusion are fixed.
See ../ for the measured valid20 result. No published upstream score is claimed for these weights.

Verify: `python verify_bundle.py`

Inference: `python infer_lite.py --input <images> --output <new_directory> --gpus 2,3,4,5,6,7`
