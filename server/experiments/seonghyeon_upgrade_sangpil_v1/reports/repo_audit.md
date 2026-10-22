# Repository audit

- Python: `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python`; exact historical environment reused, no package upgrades.
- Data: train70 / valid20 / test10, exact image/mask pairs, all opaque RGBA grayscale; strict palette validated; no identical grayscale arrays.
- Checkpoints: 100 existing last.pt files SHA-256 hashed; seed32 strict load passed.
- Source: `/Data3/sppark/komap/experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/source`. source/train.py is the actual entry, NOT vendor/train.py.
- B4 encoder channels: [3, 48, 32, 56, 160, 448]; decoder channels [256,128,64,32,16], BN, no attention; final 3x3 head16→4.
- Input: gray/255, scalar mean0.449/std0.226, copied3. OpenCV linear resize fx=fy=4000/W (OpenCV rounding), reflect right/bottom padding to32.
- Inference: BF16 forward, FP32 softmax, flip-before-rotation D4 in k0..7 order; average8 then restore native with area when enlarged, bilinear otherwise.
- Historical training: area-weighted sampling, batch2, 1536 crops, 16000 updates; AdamW lr3e-4 wd.01, 300 warmup*cosine, gradient clip1; CE+batch/class Dice(smooth1). BN unchanged, no EMA.
- Source augmentation: native pre-crop int(1536*2.1/scale), Affine(.7–1.4,±180°,p.7), reflect pad, random crop, D4, brightness/contrast(.3/.4,p.8), gamma60–160(p.5), blur/noise(p.3).
- Seed32 historical image mIoU: 0.802265456654121. Full valid20 reproduction passed; see seed32_reproduction.json.
- No checkpoint selection on valid: fixed last at16000. Validation observed during development, so scores are development results.
- Root README describes an older DINO design; current input spec and verified B4 code govern this implementation.
- GPUs0–5: six RTX4090 24GiB; independent one-GPU jobs preserve batch2. GPUs6/7 excluded.
- A-prefix overlap across train/valid: ['A125', 'A153', 'A181', 'A191', 'A40', 'A55', 'A7']; these are unconfirmed specimen hints, not proof of independent physical groups.
- Submission PNG naming will follow original stems with _mask.png; official archive requirements are not present, so no claimed competition-ready zip.
- input/README mentions SHA256SUMS and VALIDATION_REPORT files that are absent; all available input files are hashed in audit.json.
