#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KOMAP_PYTHON="${KOMAP_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${KOMAP_GPUS:-0,1,2,5,6,7}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -lt 1 ]]; then
  echo 'KOMAP_GPUS must list GPU IDs, e.g. 0,1,2,5,6,7' >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then set -- suite; fi
exec "$KOMAP_PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="${#GPU_IDS[@]}" "$EXPERIMENT_DIR/run.py" "$@"
