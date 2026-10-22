#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model333_b4_imagewise_lovasz_group_oof"
QUEUE_PID="${QUEUE_PID:-716980}"
MODEL326_RUN="$ROOT/outputs/model326_hybrid_fullstats_group_oof/run.log"
MODEL326_ROUTE="$ROOT/outputs/model326_model270_fusion/route_probe.json"
MODEL328_RUN="$ROOT/outputs/model328_b4_width4000_uniform_group_oof/run.log"
MODEL328_ROUTE="$ROOT/outputs/model328_model270_fusion/route_probe.json"
MODEL329_RUN="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/run.log"
MODEL329_ROUTE="$ROOT/outputs/model329_model270_fusion/route_probe.json"
mkdir -p "$OUT"
QUEUE_LOG="$OUT/queue.log"

echo "Waiting for the active Model326->328->329 queue (PID $QUEUE_PID)." \
  | tee -a "$QUEUE_LOG"
while kill -0 "$QUEUE_PID" 2>/dev/null; do
  sleep 60
done

if ! grep -q '^DONE:' "$MODEL326_RUN" 2>/dev/null; then
  echo "Model326 pipeline ended without its DONE marker; not starting Model333." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi

robust_80() {
  local route="$1"
  [[ -s "$route" ]] || return 1
  "$PY" - "$route" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
score = d["best_image_empty_zero"]["image_mean_miou"]
ci = d["grouped_cv_vs_model270_empty_zero"]["group_bootstrap_95ci_delta_pp"]
print(f"{score:.6%} valid; grouped-CV 95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
sys.exit(0 if score >= 0.80 and ci[0] > 0 else 1)
PY
}

for route in "$MODEL326_ROUTE" "$MODEL328_ROUTE" "$MODEL329_ROUTE"; do
  if [[ -s "$route" ]]; then
    if robust_80 "$route"; then
      echo "Robust 80% target already met by $route; Model333 is unnecessary." \
        | tee -a "$QUEUE_LOG"
      exit 0
    fi
  fi
done

if [[ -s "$MODEL328_ROUTE" ]]; then
  if ! grep -q '^DONE:' "$MODEL328_RUN" 2>/dev/null && \
     ! grep -q '^DONE:' "$MODEL329_RUN" 2>/dev/null; then
    echo "Model328/329 ended without a completed-evaluation marker; not starting Model333." \
      | tee -a "$QUEUE_LOG"
    exit 1
  fi
else
  echo "No completed Model328 route probe; refusing to infer that prior jobs succeeded." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi

if [[ ! -s "$MODEL329_ROUTE" ]] || ! grep -q '^DONE:' "$MODEL329_RUN" 2>/dev/null; then
  echo "Model329 did not complete its route evaluation; not starting Model333." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi

echo "Models326/328/329 missed the robust 80% gate; preparing paired Model333." \
  | tee -a "$QUEUE_LOG"
GPU_ID=6
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(env -u CUDA_VISIBLE_DEVICES GPU_ID="$GPU_ID" "$PY" -c \
    'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU $GPU_ID headroom check $stable/3 passed" | tee -a "$QUEUE_LOG"
  else
    stable=0
    echo "P100 GPU $GPU_ID busy; retrying in 60s" | tee -a "$QUEUE_LOG"
  fi
  if (( stable < 3 )); then sleep 60; fi
done
export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec bash "$ROOT/run_model333_b4_lovasz_group_oof.sh"
