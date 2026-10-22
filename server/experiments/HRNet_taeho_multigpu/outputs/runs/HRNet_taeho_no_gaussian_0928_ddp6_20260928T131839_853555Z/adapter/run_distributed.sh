#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${HRNET_GPUS:-0,1,2,3,4,5}"
IFS=',' read -r -a hrnet_devices <<< "$CUDA_VISIBLE_DEVICES"
declare -A hrnet_seen=()
for gpu in "${hrnet_devices[@]}"; do
  case "$gpu" in
    0|1|2|3|4|5) ;;
    *) echo 'HRNET_GPUS must list GPU indices from 0 to 5.' >&2; exit 2 ;;
  esac
  if [[ -n "${hrnet_seen[$gpu]:-}" ]]; then
    echo 'Repeated GPU index.' >&2; exit 2
  fi
  hrnet_seen[$gpu]=1
done
if [[ "${#hrnet_devices[@]}" -eq 0 || "$CUDA_VISIBLE_DEVICES" == *, ]]; then
  echo 'Empty GPU index.' >&2; exit 2
fi
hrnet_python="${HRNET_PYTHON:-$PWD/../HRNet_taeho/.venv/bin/python}"
if [[ ! -x "$hrnet_python" ]]; then
  echo 'HRNet_taeho Python environment missing; set HRNET_PYTHON.' >&2; exit 1
fi
exec "$hrnet_python" -B -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#hrnet_devices[@]}" worker.py "$@"
