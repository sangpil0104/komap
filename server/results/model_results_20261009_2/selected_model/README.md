# Model349 fusion805 — local retrained model

All 27 component weights were newly trained. Main candidate: stage1 R+E (no F); stage2 is retained for comparison.
No upstream score is claimed for these new weights. See the containing result report for measured image mIoU.

Use conda komap. Check hashes: `python verify_model.py`.
R: `python Model349_exh14/infer_lite.py --input <images> --output <new_R> --gpus 2,3,4,5,6,7`
E: `CUDA_VISIBLE_DEVICES=2 python infer_model401.py --input <images> --assets assets --output <new_E>`
Stage1: `python infer_stage1.py --release <new_R> --expert <new_E> --output <new_stage1>`
Stage2: `python infer_fusion.py --input <images> --release-labels <new_R>/predicted_labels --expert-probabilities <new_E> --refiner assets/refiner/epoch8.pt --output <new_stage2> --device cpu`
