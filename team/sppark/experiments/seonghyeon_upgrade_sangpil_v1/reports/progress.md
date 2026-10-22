# Progress

- Scope: input spec P0–P7 implementation, GPUs0–5; user starts real-data training.
- Current status: TESTED; implementation and requested pre-training preparation complete.
- Changed files: alphase/*.py, configs/*.yaml, configs/experiments/*.yaml, configs/examples/*.yaml, scripts/*.py, tests/*.py, manifests/*.jsonl, reports/*, README.md, run.sh, requirements.lock.txt, .gitignore. No input or historical source/checkpoint modifications.

| Phase | Status | Evidence |
|---|---|---|
| P0 | RUN_COMPLETED | 100 checkpoint SHA-256 manifest,70/20/10 data audit, exact seed32 valid20 regression |
| P1 | TESTED | native cache, corrupt-cache refusal, streaming mean/K1/repeated100, ensemble resume; actual seed32 cache+diagnostics completed;100-model sweep unrun |
| P2 | TESTED | original-coordinate ramps,D4,validity,relative channels, actual B4 all-off equality |
| P3 | TESTED | independent branches,zero-init,context gradient; full1536/768 batch2 GPU backward |
| P4 | TESTED | train-only isolation,mixed fallback,online hard refresh+resume,native differentiable NLL/Lovasz |
| P5 | TESTED | synthetic uninterrupted vs resumed loop exact weights/EMA/LR/sample count; real-data learning unrun |
| P6 | TESTED | strict registry+ancestry validator,5-fold orchestration,8-channel refiner; synthetic full workflow passed; real OOF/refiner training unrun |
| P7 | RUN_COMPLETED | baseline seed32 test10 native palette PNGs validated; final improved model not selected |

Executed commands (all from this experiment unless prefixed):

```bash
python scripts/audit_repository.py
python scripts/make_configs.py
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_seed32.py
./run.sh validate-config --config configs/experiments/E04.yaml
CUDA_VISIBLE_DEVICES=1 ./run.sh smoke --config configs/examples/enhanced_candidate.yaml --steps 2
CUDA_VISIBLE_DEVICES=1 ./run.sh smoke --config configs/examples/enhanced_candidate.yaml --steps 2 --full-size
python -m pytest tests -q
./run.sh build-oof --config configs/oof_train70.yaml
./run.sh launch
./run.sh cache-probs --members all
CUDA_VISIBLE_DEVICES=2 ./run.sh cache-member --member seed_32 --execute
./run.sh diagnose --prediction-manifest outputs/probabilities/seed_32/manifest.jsonl --output reports/seed32_baseline
python scripts/augmentation_audit.py
CUDA_VISIBLE_DEVICES=3 ./run.sh predict --config configs/final.yaml --split test --execute
```

`python` above means `/Data3/sppark/komap/experiments/seonghyeon_seed10_v1/.venv/bin/python`. Dry-run launch/cache/all-OOF commands did not train or start the100-model sweep. GPU smoke used synthetic inputs,0 optimizer updates; head was perturbed only in memory to check earlier-branch gradients. Unit tests optimized small synthetic models/arrays only, never the real project train images.

Known limits / not executed:

- Real train-patch overfit, actual4000/16000-update learning,100-model inference ensemble,fold teacher learning,refiner learning,outer-fold evaluation remain unrun by design.
- No accuracy improvement is claimed. Seed32 reproduction80.2265456654% is the existing baseline.
- Historical six-worker augmentation random stream is not reproduced by new indexed RNG; historical inference is exactly reproduced.
- Hard mining can add inference time; score maps/age are checkpointed and no validation images are used.
- Same validation20 reused for model selection yields development results, not independent outer-test performance.
- Specimen groups/official submission archive rules are unavailable. A-prefix overlap is reported, PNG naming is documented.
- Test10 outputs are seed32 baseline masks, not new model outputs; test ground truth is unavailable.

Next action: user may start the12 paired experiments on GPUs0–5 with `./run.sh launch --execute`. No training process is running.

Final test result: 44 passed in 5.38s. Final preservation check: 306 files unchanged. See ../VALIDATION_REPORT.md.
