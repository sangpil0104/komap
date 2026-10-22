# Changes to the vendored training code

`experiments/` is the code of `Model349_train_route_code.zip` (CNUKoMaP/lhs). One line differs:

| File | Change | Why |
|---|---|---|
| `experiments/model322_team_b4/KoMaP/train.py` | the DataLoader worker seed no longer adds `int(time.time())` | the wall-clock term made every run draw different crops even with the same `--seed` |

Other nondeterminism is handled without editing files:
* B4 326/329 use their existing `--deterministic_workers` flag.
* `oof_design/runtime/sitecustomize.py` (first on `PYTHONPATH`) turns cuDNN autotuning off, prefers
  deterministic kernels, and ignores later `cudnn.benchmark = True` assignments in historical scripts.
