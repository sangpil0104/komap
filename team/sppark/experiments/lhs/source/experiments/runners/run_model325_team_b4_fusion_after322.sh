#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
B4_OUT="$ROOT/outputs/model322_team_b4_group_oof"
OUT="$ROOT/outputs/model325_team_ours_fusion"
RESULT="$OUT/route_probe.json"
RUN_PID=654330

echo "Waiting for Model322 grouped-OOF run PID $RUN_PID to finish." >&2
while kill -0 "$RUN_PID" 2>/dev/null; do
  if grep -q '^DONE:' "$B4_OUT/run.log" 2>/dev/null; then
    break
  fi
  sleep 60
done

if ! grep -q '^DONE:' "$B4_OUT/run.log" 2>/dev/null; then
  echo "Model322 exited without its DONE marker; refusing to launch Model325." >&2
  exit 1
fi

CANDIDATE="$B4_OUT/valid_ensemble/probabilities.npz"
for required in \
  "$CANDIDATE" \
  "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
  "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
  "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
  "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz"; do
  if [[ ! -s "$required" ]]; then
    echo "Required Model325 input missing/empty: $required" >&2
    exit 1
  fi
done

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite existing Model325 result: $RESULT" >&2
  exit 1
fi

mkdir -p "$OUT"
cd "$ROOT"
echo "Launching Model325: Model322 team-B4 reproduction + Model270." >&2
CUDA_VISIBLE_DEVICES= nice -n 19 "$PY" model323_model270_branch_route_probe.py \
  --entries "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
  --model195 "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
  --tent20 "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
  --model269 "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
  --candidate "$CANDIDATE" \
  --candidate-id team_b4_model322 \
  --candidate-order team \
  --output "$RESULT"

echo "Model325 complete: $RESULT" >&2
