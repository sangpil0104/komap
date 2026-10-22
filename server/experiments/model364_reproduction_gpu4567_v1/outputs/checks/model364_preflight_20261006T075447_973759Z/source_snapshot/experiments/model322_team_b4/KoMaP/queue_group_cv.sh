#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model322_team_b4_group_oof"
CODE="$ROOT/model322_team_b4/KoMaP"
GPU_ID="${GPU_ID:-0}"
mkdir -p "$OUT"

if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an index from 0 to 7 (got: $GPU_ID)" >&2
  exit 2
fi

# B4 at crop1536/bs2 peaked at 13.2 GiB in the isolated smoke run.
# Require >14.5 GiB free for three consecutive checks to avoid fighting
# another user's P100 job or starting into transient free memory.
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(GPU_ID="$GPU_ID" "$PY" -c 'import os, torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU ${GPU_ID} headroom check ${stable}/3 passed" >> "$OUT/queue.log"
  else
    stable=0
    echo "P100 GPU ${GPU_ID} headroom insufficient; retrying in 60s" >> "$OUT/queue.log"
  fi
  if (( stable < 3 )); then sleep 60; fi
done

echo "Stable memory headroom confirmed on P100 GPU ${GPU_ID}; starting grouped CV" >> "$OUT/queue.log"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec bash "$CODE/run_group_cv.sh"
