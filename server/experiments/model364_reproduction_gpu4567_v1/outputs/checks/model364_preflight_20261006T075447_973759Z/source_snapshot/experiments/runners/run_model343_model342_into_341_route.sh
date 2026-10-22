#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model343_model342_into_341_route"

for required in \
  "$ROOT/model343_model342_into_341_route.py" \
  "$ROOT/model341_model329_joint_route_search.py" \
  "$ROOT/model335_exhaustive_b4_route_search.py" \
  "$ROOT/model198_fast_branch_search.py" \
  "$ROOT/outputs/model339_joint_gamma_route_search/results.json" \
  "$ROOT/outputs/model341_model329_joint_route_search/results.json" \
  "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
  "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
  "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
  "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
  "$ROOT/outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz" \
  "$ROOT/outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz" \
  "$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz"; do
  [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

if [[ -e "$OUT/results.json" ]]; then
  echo "Model343 result already exists; refusing to overwrite it." >&2
  exit 1
fi

mkdir -p "$OUT"
echo "Starting low-priority CPU-only Model343 search (625 routes; grouped specimen holdout)."
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
  "$ROOT/model343_model342_into_341_route.py" \
  --output "$OUT/results.json" 2>&1 | tee "$OUT/run.log"

echo "DONE: Model343 results at $OUT/results.json"
