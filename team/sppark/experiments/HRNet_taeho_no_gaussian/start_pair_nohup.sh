#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
baseline_dir="$PWD/../HRNet_taeho"
baseline_gpu="${BASELINE_GPU:-0}"
ablation_gpu="${ABLATION_GPU:-1}"
for gpu in "$baseline_gpu" "$ablation_gpu"; do
  case "$gpu" in
    0|1|2|3|4|5) ;;
    *) echo 'GPU indices must be from 0 to 5.' >&2; exit 2 ;;
  esac
done
if [[ "$baseline_gpu" == "$ablation_gpu" ]]; then
  echo 'Choose different GPUs for the two experiments.' >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--dry-run' ) ]]; then
  echo 'Usage: start_pair_nohup.sh [--dry-run]' >&2
  exit 2
fi
echo "HRNet_taeho: physical GPU $baseline_gpu"
echo "HRNet_taeho_no_gaussian: physical GPU $ablation_gpu"
if [[ "${1:-}" == '--dry-run' ]]; then
  echo 'Dry-run: no training processes started.'
  exit 0
fi
HRNET_GPU="$baseline_gpu" "$baseline_dir/start_nohup.sh"
HRNET_GPU="$ablation_gpu" ./start_nohup.sh
