#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--dry-run' ) ]]; then
  echo 'Usage: run_sequence.sh [--dry-run]' >&2; exit 2
fi
echo "GPU allocation for each stage: ${HRNET_GPUS:-0,1,2,3,4,5}"
echo 'Stage 1: HRNet_taeho training and validation'
echo 'Stage 2: HRNet_taeho_no_gaussian training and validation'
if [[ "${1:-}" == '--dry-run' ]]; then
  echo 'Dry-run: no training started.'
  exit 0
fi
./run_distributed.sh train --variant baseline --execute
echo 'Stage 1 completed; starting stage 2.'
./run_distributed.sh train --variant no_gaussian --execute
echo 'Both experiments completed.'
