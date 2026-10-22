#!/bin/bash
# Alloy-grouped 5-fold training, out-of-fold evaluation, and a 5-model
# ensemble prediction of the test split (the submission recipe).
# Usage: scripts/kfold.sh <data-root> [runs-root]
set -euo pipefail
DATA_ROOT=${1:?usage: scripts/kfold.sh <data-root> [runs-root]}
RUNS=${2:-runs/kfold}
FOLDS=$RUNS/folds
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

uv run python make_folds.py --data-root "$DATA_ROOT" --output-root "$FOLDS"
for i in 0 1 2 3 4; do
  mkdir -p "$RUNS/fold$i"
  uv run python train.py --data-root "$FOLDS/fold$i" --output-dir "$RUNS/fold$i" | tee "$RUNS/fold$i/train.log"
done
uv run python kfold_evaluate.py --folds-root "$FOLDS" --runs-root "$RUNS" | tee "$RUNS/oof.log"
uv run python predict.py --checkpoint "$RUNS"/fold{0,1,2,3,4}/last.pt \
    --images-dir "$DATA_ROOT/test/images" --output-dir "$RUNS/test_predictions"
