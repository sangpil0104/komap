#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
MODEL322_OUT="$ROOT/outputs/model322_team_b4_group_oof"
MODEL325_OUT="$ROOT/outputs/model325_team_ours_fusion/route_probe.json"
MODEL325_PID="${MODEL325_PID:-}"
LOG="$ROOT/outputs/model326_hybrid_fullstats_group_oof/queue.log"
mkdir -p "$(dirname "$LOG")"

echo "Waiting for Model322 and Model325 fusion to finish before the hybrid-training experiment." >&2
while ! grep -q '^DONE:' "$MODEL322_OUT/run.log" 2>/dev/null; do
  echo "Model322 grouped-OOF run is still active; checking again in 60s." >> "$LOG"
  sleep 60
done
while [[ ! -s "$MODEL325_OUT" ]]; do
  if [[ -z "$MODEL325_PID" ]] || ! kill -0 "$MODEL325_PID" 2>/dev/null; then
    echo "Model325 result is missing and its PID is absent/dead; refusing Model326." >&2
    exit 1
  fi
  echo "Model325 fusion still running; checking again in 30s." >> "$LOG"
  sleep 30
done
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("candidate_count")==32, d.get("candidate_count"); assert "grouped_cv_vs_model270_empty_zero" in d' "$MODEL325_OUT"

echo "Model325 fusion is complete; proceeding with Model326 hybrid training regardless of score." | tee -a "$LOG"

# Shared-GPU policy: reserve exactly one P100 only after stable free-memory checks.
GPU_ID=6
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(GPU_ID="$GPU_ID" "$PY" -c 'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU $GPU_ID headroom check $stable/3 passed" | tee -a "$LOG"
  else
    stable=0
    echo "P100 GPU $GPU_ID busy; retrying in 60s" | tee -a "$LOG"
  fi
  if (( stable < 3 )); then sleep 60; fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec bash "$ROOT/run_model326_hybrid_fullstats_group_oof.sh"
