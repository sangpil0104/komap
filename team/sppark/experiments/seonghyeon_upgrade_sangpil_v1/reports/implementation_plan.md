# Implementation map

All new source/config/manifests/reports live outside input/ in this experiment. Historical source/checkpoints are read-only. The user starts every real-data training job; long inference and training commands default to dry-run.

| Spec phase | Files | Commands |
|---|---|---|
| P0 audit + exact baseline | scripts/audit_repository.py, contracts.py, legacy.py, config.py | audit, validate-config, reproduce |
| P1 probabilities + diagnostics | cache.py, evaluation.py, jobs.py | cache-probs, cache-member, ensemble, diagnose |
| P2 native geometry + texture | geometry.py, data.py, models.py | E02, E03b, E04 profiles |
| P3 context residual fusion | models.py, tests/test_pipeline.py | E05, smoke |
| P4 train sampler + native loss | data.py, losses.py | E06, E07, mine-hard |
| P5 recipes / EMA / resume | training.py, jobs.py, tests/test_resume.py | train, launch, evaluate |
| P6 strict OOF + refiner | oof.py, refiner.py | build-oof, validate-oof, train-refiner |
| P7 artifacts + native PNGs | evaluation.py, cli.py | predict, validate-submission |

The suggested nested modules are consolidated into a small package without changing public command names. Extra commands: launch (paired seeds on six GPUs), cache-member (one GPU worker), mine-hard (explicit in-sample train mining). No servers are started.

Recipe decisions:

- Legacy inference is independently implemented and compared with read-only original common.py. It keeps BF16, scalar gray normalization, OpenCV resize, flip-before-rotation D4, and native area restoration.
- New geometry uses uniform train-image selection, indexed deterministic sampling, shared D4, photometric transform before relative features. Arbitrary rotation/scale jitter are disabled. The historical operator sequence remains available in legacy_scaled_crop; indexed RNG changes its stream versus the historical six-worker stream, so fresh training is not claimed bitwise identical to an old run.
- Independent branches use GN; only the last residual convolution is zero initialized. Disabled modules are absent. Context feature D4 is undone for native alignment and then reapplied.
- Finetuning compares each seed with its original matching parent, not the previous experiment's selected checkpoint. Plain: 4000 base updates. Residual models: 500 frozen-base updates +3500 joint updates with a new stage-B optimizer/scheduler.
- Hard mining: random fallback before update500, then on-demand native predictions of the selected TRAIN image. Refresh after1000 updates, no D4 by default. Pixel NLL clipped at that image's95th percentile, Al3Ni→EutecticSi weight2. Score maps and update ages are included in resume state. An external strict OOF or explicitly permitted in-sample train cache can replace online mining.
- Validation: full valid20 D4 every1000 updates and final; best raw/EMA stored separately. EMA parameters averaged after successful optimizer updates, all BN buffers copied. BF16 needs no GradScaler; nonfinite loss/gradients fail before update.
- OOF: five image-level folds, one seed per fold initially, ImageNet-only initialization, fixed-update teacher checkpoints, plus matching full-train deployment teacher. Known specimen groups are unavailable; A-prefix overlap is diagnostic, not certified group separation.
- Refiner: native full-frame inference, native512 train crops, 8 input channels, four-class residual logits, NLL+0.5 Lovasz. OOF/deployment profile and ensemble size must match. No outer-fold claim is made.
- Final default remains the verified seed32 baseline until development experiments select a new configuration. E09/E10 gains and E11 heterogeneous combinations cannot be selected before actual training.
- Submission naming is image ID + _mask.png; only PNG native-size/palette/file-set validation is claimed because an official archive convention was not supplied.
